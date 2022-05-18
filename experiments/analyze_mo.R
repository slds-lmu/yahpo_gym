library(data.table)  # 1.14.2
library(ggplot2)  # 3.3.5
library(pammtools)  # 0.5.7
library(emoa)  # 0.5-0.1
library(mlr3misc)  # 0.10.0

values = c("#386cb0", "#fdb462", "#7fc97f", "#ef3b2c", "#662506", "#a6cee3", "#984ea3")
results = readRDS("results/results_mo.rds")
results[method == "randomx4", budget := 1 / 4, by = .(method, scenario, instance, targets, repl)]
results[method %nin% "randomx4", budget := 1, by = .(method, scenario, instance, targets, repl)]
results[, cumbudget := 0.0]
results[, cumbudget := as.double(cumsum(budget)), by = .(method, scenario, instance, targets, repl)]
results[, cumbudget_scaled := cumbudget / max(cumbudget), by = .(method, scenario, instance, targets, repl)]

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
            map_dtr(seq(0, 1, length.out = 101), function(budget_) {
              dat = unique(tmp[method == method_ & repl == repl_ & cumbudget_scaled <= budget_, y_cols, with = FALSE])
              if (nrow(dat) == 0L) return(data.table())
              pareto = nondominated_points(t(unique(dat)))
              hvi = hypervolume_indicator(pareto, o = pareto_ref, ref = nadir)
              data.table(scenario = scenario_, targets = targets_, instance = instance_, method = method_, repl = repl_, cumbudget_scaled = budget_, hvi = hvi)
            })
          })
        })
      })
   })
})

agg = dat[, .(mean_hvi = mean(hvi), se_hvi = sd(hvi) / sqrt(.N)), by = .(cumbudget_scaled, method, scenario, instance, targets)]
agg[, method := factor(method, levels = c("random", "randomx4", "parego", "smsego", "ehvi", "mego", "mies"), labels = c("Random", "Random x4", "ParEGO", "SMS-EGO", "EHVI", "MEGO", "MIES"))]

g = ggplot(aes(x = cumbudget_scaled, y = mean_hvi, colour = method, fill = method), data = agg[cumbudget_scaled > 0.10]) +
  scale_y_log10() +
  geom_step() +
  geom_stepribbon(aes(min = mean_hvi - se_hvi, max = mean_hvi + se_hvi), colour = NA, alpha = 0.3) +
  scale_colour_manual(values = values) +
  scale_fill_manual(values = values) +
  labs(x = "Fraction of Budget Used", y = "Mean Normalized HVI", colour = "Optimizer", fill = "Optimizer") +
  facet_wrap(~ scenario + instance + targets, scales = "free") +
  theme_minimal() +
  theme(legend.position = "bottom", legend.title = element_text(size = rel(0.75)), legend.text = element_text(size = rel(0.75)))
ggsave("plots/anytime_mo.png", plot = g, device = "png", width = 12, height = 15)

methods = unique(agg$method)
ranks = map_dtr(unique(agg$scenario), function(scenario_) {
  map_dtr(unique(agg$instance), function(instance_) {
    map_dtr(unique(agg$targets), function(targets_) {
      map_dtr(unique(agg$cumbudget_scaled), function(cumbudget_scaled_) {
        res = agg[scenario == scenario_ & instance == instance_ & targets == targets_ & cumbudget_scaled == cumbudget_scaled_]
        if (nrow(res) == 0L) {
          return(data.table())
        }
        setorderv(res, "mean_hvi")
        data.table(rank = match(methods, res$method), method = methods, scenario = scenario_, instance = instance_, targets = targets_, cumbudget_scaled = cumbudget_scaled_)
      })
    })
  })
})

ranks_overall = ranks[, .(mean = mean(rank, na.rm = TRUE), se = sd(rank, na.rm = TRUE) / sqrt(sum(!is.na(rank)))), by = .(method, cumbudget_scaled)]

g = ggplot(aes(x = cumbudget_scaled, y = mean, colour = method, fill = method), data = ranks_overall[cumbudget_scaled > 0.10]) +
  geom_line() +
  geom_ribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3) +
  scale_colour_manual(values = values) +
  scale_fill_manual(values = values) +
  labs(x = "Fraction of Budget Used", y = "Mean Rank", colour = "Optimizer", fill = "Optimizer") +
  theme_minimal() +
  theme(legend.position = "bottom", legend.title = element_text(size = rel(0.75)), legend.text = element_text(size = rel(0.75)))
ggsave("plots/anytime_average_rank_mo.png", plot = g, device = "png", width = 6, height = 4)

library(scmamp)  # 0.3.2
best_agg = agg[cumbudget_scaled == 0.25]
best_agg[, problem := paste0(scenario, "_", instance, "_", targets)]
tmp = - as.matrix(dcast(best_agg, problem ~ method, value.var = "mean_hvi")[, -1])
friedmanTest(tmp) # 0.25: chi(6) 48.103, p < 0.001
png("plots/cd_025_mo.png", width = 6, height = 4, units = "in", res = 300, pointsize = 10)
plotCD(tmp, cex = 1)
dev.off()

best_agg = agg[cumbudget_scaled == 1]
best_agg[, problem := paste0(scenario, "_", instance, "_", targets)]
tmp = - as.matrix(dcast(best_agg, problem ~ method, value.var = "mean_hvi")[, -1])
friedmanTest(tmp) # 1: chi(6) 41.091,, p < 0.001
png("plots/cd_1_mo.png", width = 6, height = 4, units = "in", res = 300, pointsize = 10)
plotCD(tmp, cex = 1)
dev.off()

