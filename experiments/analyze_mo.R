library(data.table)
library(mlr3misc)
library(ggplot2)
library(pammtools)
library(emoa)
results = readRDS("results_mo.rds")

dat = map_dtr(unique(results$scenario), function(scenario_) {
  map_dtr(unique(results$instance), function(instance_) {
    map_dtr(unique(results$targets), function(targets_) {
      tmp = results[scenario == scenario_ & instance == instance_ & targets == targets_]
      y_cols = if (targets_ == "val_accuracy_val_cross_entropy") {
        c("val_accuracy", "val_cross_entropy")
      } else {
        strsplit(targets_, split = "_")[[1L]]
      }
      tmp[, (y_cols) := lapply(.SD, function(x) (x - min(x)) / (max(x) - min(x))), .SDcols = y_cols]
      nadir = t(t(apply(tmp[, y_cols, with = FALSE], MARGIN = 2L, FUN = max) + 1L))
      pareto_ref = nondominated_points(t(unique(tmp[, y_cols, with = FALSE])))
        map_dtr(unique(tmp$method), function(method_) {
          map_dtr(unique(tmp$repl), function(repl_) {
            map_dtr(unique(tmp$iter), function(iter_) {
              dat = unique(tmp[method == method_ & repl == repl_ & iter <= iter_, y_cols, with = FALSE])
              if (nrow(dat) == 0L) return(data.table())
              pareto = nondominated_points(t(unique(dat)))
              dhv = dominated_hypervolume(pareto, ref = nadir)
              hvi = hypervolume_indicator(pareto, o = pareto_ref, ref = nadir)
              eps = epsilon_indicator(pareto, o = pareto_ref)
              data.table(scenario = scenario_, targets = targets_, instance = instance_, method = method_, repl = repl_, iter = iter_, dhv = dhv, hvi = hvi, eps = eps)
            })
          })
        })
      })
   })
})

agg = dat[, .(mean_dhv = mean(dhv), se_dhv = sd(dhv) / sqrt(.N), mean_hvi = mean(hvi), se_hvi = sd(hvi) / sqrt(.N), mean_eps = mean(eps), se_eps = sd(eps) / sqrt(.N)), by = .(scenario, instance, targets, method, iter)]

ggplot(aes(x = iter, y = mean_hvi, colour = method, fill = method), data = agg) +
  geom_step(aes(x = iter, y = mean_hvi)) +
  geom_stepribbon(aes(x = iter, ymin = mean_hvi - se_hvi, ymax = mean_hvi + se_hvi), colour = NA, alpha = 0.5) +
  xlab("Iteration") +
  ylab("Mean Hypervolume Indicator") +
  labs(colour = "Optimizer", fill = "Optimizer") +
  theme_minimal(base_size = 10) +
  facet_wrap(~ scenario + instance + targets, scales = "free")

methods = unique(dat$method)
ranks = map_dtr(unique(dat$scenario), function(scenario_) {
  map_dtr(unique(dat$instance), function(instance_) {
    map_dtr(unique(dat$targets), function(targets_) {
      map_dtr(unique(dat$repl), function(repl_) {
        map_dtr(unique(dat$iter), function(iter_) {
          res = dat[scenario == scenario_ & targets == targets_ & instance == instance_ & repl == repl_ & iter == iter_]
          if (nrow(res) == 0L) {
            return(data.table())
          }
          setorderv(res, "hvi", 1)
          data.table(rank = match(methods, res$method), method = methods, scenario = scenario_, instance = instance_, targets = targets_, repl = repl_, iter = iter_)
        })
      })
    })
  })
})

ranks_agg = ranks[, .(mean = mean(rank), se = sd(rank) / sqrt(.N)), by = .(method, scenario, instance, targets, iter)]

ggplot(aes(x = iter, y = mean, colour = method, fill = method), data = ranks_agg) +
  geom_line(lwd = 1) +
  geom_ribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3) +
  facet_wrap(~ scenario + instance + targets, scales = "free")

ranks_total = ranks_agg[, .(mean = mean(mean, na.rm = TRUE), se = sd(mean) / sqrt(.N)), by = .(method, iter)]

ggplot(aes(x = iter, y = mean, colour = method, fill = method), data = ranks_total) +
  geom_line(lwd = 1) +
  geom_ribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3)

