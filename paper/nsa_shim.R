## ---------------------------------------------------------------------------
## Backend shim: the paper's original call signatures, current implementation.
##
## The manuscript was written against ANTsR's v1 NSA-Flow, whose machinery
## (soft/polar retraction, learning-rate strategies, optimizer backends,
## warm-up, aggression) no longer exists.  v2 replaces all of it with one energy
## and a spectral projected gradient carrying a stationarity certificate.  This
## file keeps the paper's function names and argument lists so the experiment
## chunks are unchanged, and routes them to the Python package.
##
## Arguments that named v1-only machinery are accepted and ignored; each is
## listed in `nsa_shim_ignored()` so the document can report exactly what was
## dropped rather than silently discarding it.
## ---------------------------------------------------------------------------
library(reticulate)

.nsa <- NULL
.np <- NULL
.torch <- NULL
.nsa_ignored <- character(0)

nsa_shim_init <- function(repo = normalizePath(file.path(".."))) {
  Sys.setenv(PYTHONPATH = repo)
  .nsa   <<- import("nsa_flow")
  .np    <<- import("numpy")
  .torch <<- import("torch")
  invisible(.nsa$`__version__`)
}

.note_ignored <- function(...) {
  nm <- c(...)
  .nsa_ignored <<- sort(unique(c(.nsa_ignored, nm)))
}

nsa_shim_ignored <- function() .nsa_ignored

## numpy for the solver entry points, torch for the tensor-level helpers
.as_mat <- function(M) .np$array(as.matrix(M), dtype = "float64")
.as_t   <- function(M) .torch$as_tensor(.np$array(as.matrix(M), dtype = "float64"))

## --- the refinement solver -------------------------------------------------
## v1: nsa_flow_autograd(Y0, X0, w, retraction, lr_strategy, optimizer, ...)
## v2: minimise (1-w)||Y - X0||_F^2/||X0||_F^2 + w Dtilde(Y) over Y >= 0
## `fidelity` picks what "close to the target" means.  "auto" uses the
## entrywise distance for a non-negative target and the sign-blind subspace
## distance for a signed one, because the entrywise distance charges a
## non-negative Y for negative entries of X0 it cannot reach and so degenerates
## toward max(0, X0).  The choice made, the target's negative mass and the
## distance from max(0, X0) are all returned.
nsa_flow_autograd <- function(Y0, X0 = NULL, w = 0.5, max_iter = 1000,
                              tol = 1e-8, verbose = FALSE, apply_nonneg = TRUE,
                              seed = 42, plot = TRUE, fidelity = "auto", ...) {
  dots <- list(...)
  if (length(dots)) .note_ignored(names(dots))
  target <- if (is.null(X0)) Y0 else X0
  r <- suppressWarnings(.nsa$nsa_flow(
                     .as_mat(target), w = w, init = .as_mat(Y0),
                     nonneg = isTRUE(apply_nonneg), fidelity = fidelity,
                     max_iter = as.integer(max_iter), tol = tol,
                     verbose = isTRUE(verbose), keep_trace = isTRUE(plot)))
  ## A real trace plot.  Returning NULL here would be worse than it looks:
  ## `NULL + labs(...)` evaluates to NULL without error, and assigning NULL to a
  ## list element removes it, so a caller building a list of plots ends up with
  ## an empty list and the figure silently disappears.
  gg <- NULL
  if (isTRUE(plot) && !is.null(r$trace) && length(r$trace) > 0) {
    tr <- do.call(rbind, lapply(r$trace, function(z) data.frame(
      iter = as.numeric(z$iter), fidelity = as.numeric(z$fidelity),
      defect = as.numeric(z$defect))))
    long <- rbind(
      data.frame(iter = tr$iter, value = tr$fidelity, term = "fidelity"),
      data.frame(iter = tr$iter, value = pmax(tr$defect, 1e-16),
                 term = "orthogonality defect"))
    gg <- ggplot2::ggplot(long, ggplot2::aes(iter, value, colour = term)) +
      ggplot2::geom_line(linewidth = 0.7) +
      ggplot2::scale_y_log10() +
      ggplot2::labs(x = "iteration", y = NULL, colour = NULL) +
      ggplot2::theme_minimal(base_size = 9) +
      ggplot2::theme(legend.position = "bottom")
  }
  list(Y = py_to_r(r$Y$numpy()), energy = r$energy, fidelity = r$fidelity,
       defect = r$defect, iters = r$iters, converged = r$converged,
       stop_reason = r$stop_reason, grad_map = r$grad_map,
       effective_rank = r$effective_rank, trace = r$trace, plot = gg,
       fidelity_mode = r[["fidelity_mode"]],
       target_negative_mass = r[["target_negative_mass"]],
       clamp_distance = r[["clamp_distance"]])
}

## --- one-shot retraction ---------------------------------------------------
## The v1 "soft polar" retraction blended Y toward its polar factor with
## strength omega.  v2's projection onto the scaled Stiefel manifold is the
## correct Euclidean projection, (sum sigma_i / k) U V', and blending against it
## makes omega an exact blend fraction -- v1 blended against a unit-norm polar
## factor, so its realised fraction depended on p.
nsa_flow_retract <- function(Y, omega = 1.0, type = c("soft_polar", "polar")) {
  type <- match.arg(type)
  P <- py_to_r(.nsa$project_scaled_stiefel(.as_t(Y))$numpy())
  if (type == "polar") P else (1 - omega) * as.matrix(Y) + omega * P
}

## --- orthogonality measures ------------------------------------------------
## v1's functional, retained so the paper can show what it does and does not see
invariant_orthogonality_defect <- function(Y) {
  Y <- as.matrix(Y)
  G <- crossprod(Y)
  sum((G - diag(diag(G)))^2) / (sum(Y^2)^2)
}
## v2: orthoNORMality, and orthogonality
stiefel_defect <- function(Y) as.numeric(.nsa$stiefel_defect(.as_t(Y))$item())
angle_defect   <- function(Y) as.numeric(.nsa$angle_defect(.as_t(Y))$item())

## --- sparse PCA ------------------------------------------------------------
## v1: nsa_flow_pca(X, k, lambda, alpha, max_iter, nsa_w, proximal_type, ...)
## v2: the data-anchored form IS sparse PCA via NSA-Flow, with one parameter:
##     minimise (1-w)||X - X V V'||_F^2/||X||_F^2 + w C(V) over V >= 0
nsa_flow_pca <- function(X, k, nsa_w = 0.5, max_iter = 2000L, tol = NULL,
                         proximal_type = c("nsa_flow", "basic"), ...) {
  proximal_type <- match.arg(proximal_type)
  dots <- list(...)
  if (length(dots)) .note_ignored(names(dots))
  if (proximal_type == "basic") {
    ## the soft-thresholding comparator: unconstrained loadings, then threshold
    s <- svd(scale(as.matrix(X), center = TRUE, scale = FALSE), nu = 0, nv = k)
    V <- s$v
    thr <- stats::quantile(abs(V), 0.5)
    V[abs(V) < thr] <- 0
    return(list(Y = V, method = "soft-threshold"))
  }
  r <- .nsa$nsa_flow_data(.as_mat(X), k = as.integer(k), w = nsa_w,
                          max_iter = as.integer(max_iter))
  list(Y = py_to_r(r$Y$numpy()), recon = r$fidelity, defect = r$defect,
       iters = r$iters, converged = r$converged, stop_reason = r$stop_reason,
       matrix_free = r$matrix_free, method = "nsa-flow (data-anchored)")
}

## --- data-anchored basis, exposed under its own name too -------------------
nsa_flow_data <- function(X, k, w = 0.5, max_iter = 2000L, init = "relax") {
  r <- .nsa$nsa_flow_data(.as_mat(X), k = as.integer(k), w = w,
                          max_iter = as.integer(max_iter), init = init)
  list(Y = py_to_r(r$Y$numpy()), recon = r$fidelity, defect = r$defect,
       iters = r$iters, converged = r$converged, stop_reason = r$stop_reason)
}
