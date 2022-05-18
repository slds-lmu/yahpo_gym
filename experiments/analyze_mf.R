library(data.table)  # 1.14.2
library(ggplot2)  # 3.3.5
library(pammtools)  # 0.5.7
library(mlr3misc)  # 0.10.0

values = c("#386cb0", "#fdb462", "#7fc97f", "#ef3b2c", "#662506", "#a6cee3", "#984ea3")
dat = readRDS("results/results_mf.rds")
dat[, cumbudget := cumsum(budget), by = .(method, scenario, instance, repl)]
dat[, cumbudget_scaled := cumbudget / max(cumbudget), by = .(method, scenario, instance, repl)]
dat[, normalized_regret := (target - min(target)) / (max(target) - min(target)), by = .(scenario, instance)]
dat[, incumbent := cummin(normalized_regret), by = .(method, scenario, instance, repl)]

get_incumbent_cumbudget = function(incumbent, cumbudget_scaled) {
  budgets = seq(0, 1, length.out = 101)
  map_dbl(budgets, function(budget) {
    ind = which(cumbudget_scaled <= budget)
    if (length(ind) == 0L) {
      max(incumbent)
    } else {
      min(incumbent[ind])
    }
  })
}

dat_budget = dat[, .(incumbent_budget = get_incumbent_cumbudget(incumbent, cumbudget_scaled), cumbudget_scaled = seq(0, 1, length.out = 101)), by = .(method, scenario, instance, repl)]

ecdf_dat = copy(dat)
ecdf_res = 
  map_dtr(unique(ecdf_dat$method), function(method_) {
  map_dtr(unique(ecdf_dat$scenario), function(scenario_) {
  map_dtr(unique(ecdf_dat$instance), function(instance_) {
    tmp = dat[scenario == scenario_ & instance == instance_ & method == method_]
    if (NROW(tmp) == 0L) return(data.table())
    x = -tmp$target
    if (any(x > 1)) x = x/100
    x = 1 - x
    ecdf_fun = ecdf(x)
    data.table(nr = seq(min(x), max(x), length.out = 101), ecdf = ecdf_fun(seq(min(x), max(x), length.out = 101)), scenario = scenario_, instance = instance_, method = method_)
  })
  })
})
ecdf_res[, problem := paste0(scenario, ":", instance)]
ecdf_res[, method := factor(method, levels = c("random", "smac4hpo", "hb", "bohb", "dehb", "smac4mf", "optuna"), labels = c("Random", "SMAC", "HB", "BOHB", "DEHB", "SMAC-HB", "optuna"))]

g = ggplot(aes(x = nr, y = ecdf, colour = method), data = ecdf_res) +
  geom_line() +
  labs(x = "Missclassification Error", y = expression("P(X" >= "  x)"), colour = "Optimizer") +
  theme_minimal() +
  scale_colour_manual(values = values) +
  theme(legend.position = "bottom", legend.title = element_text(size = rel(0.75)), legend.text = element_text(size = rel(0.75))) +
  facet_wrap(~ problem, scales = "free", ncol = 4) +
  scale_x_reverse()

ggsave("plots/ecdf.png", plot = g, device = "png", width = 9, height = 11.25, scale = 1.2)

agg_budget = dat_budget[, .(mean = mean(incumbent_budget), se = sd(incumbent_budget) / sqrt(.N)), by = .(cumbudget_scaled, method, scenario, instance)]
agg_budget[, method := factor(method, levels = c("random", "smac4hpo", "hb", "bohb", "dehb", "smac4mf", "optuna"), labels = c("Random", "SMAC", "HB", "BOHB", "DEHB", "SMAC-HB", "optuna"))]

g = ggplot(aes(x = cumbudget_scaled, y = mean, colour = method, fill = method), data = agg_budget[cumbudget_scaled > 0.10]) +
  scale_y_log10() +
  geom_step() +
  geom_stepribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3) +
  scale_colour_manual(values = values) +
  scale_fill_manual(values = values) +
  labs(x = "Fraction of Budget Used", y = "Mean Normalized Regret", colour = "Optimizer", fill = "Optimizer") +
  facet_wrap(~ scenario + instance, scales = "free", ncol = 4) +
  theme_minimal() +
  theme(legend.position = "bottom", legend.title = element_text(size = rel(0.75)), legend.text = element_text(size = rel(0.75)))
ggsave("plots/anytime_mf.png", plot = g, device = "png", width = 12, height = 15)

overall_budget = agg_budget[, .(mean = mean(mean), se = sd(mean) / sqrt(.N)), by = .(method, cumbudget_scaled)]

g = ggplot(aes(x = cumbudget_scaled, y = mean, colour = method, fill = method), data = overall_budget[cumbudget_scaled > 0.10]) +
  scale_y_log10() +
  geom_step() +
  geom_stepribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3) +
  scale_colour_manual(values = values) +
  scale_fill_manual(values = values) +
  labs(x = "Fraction of Budget Used", y = "Mean Normalized Regret", colour = "Optimizer", fill = "Optimizer") +
  theme_minimal() +
  theme(legend.position = "bottom", legend.title = element_text(size = rel(0.75)), legend.text = element_text(size = rel(0.75)))
ggsave("plots/anytime_average_mf.png", plot = g, device = "png", width = 6, height = 4)

methods = unique(agg_budget$method)
ranks = map_dtr(unique(agg_budget$scenario), function(scenario_) {
  map_dtr(unique(agg_budget$instance), function(instance_) {
    map_dtr(unique(agg_budget$cumbudget_scaled), function(cumbudget_scaled_) {
      res = agg_budget[scenario == scenario_ & instance == instance_ & cumbudget_scaled == cumbudget_scaled_]
      if (nrow(res) == 0L) {
        return(data.table())
      }
      setorderv(res, "mean")
      data.table(rank = match(methods, res$method), method = methods, scenario = scenario_, instance = instance_, cumbudget_scaled = cumbudget_scaled_)
    })
  })
})

ranks_overall = ranks[, .(mean = mean(rank), se = sd(rank) / sqrt(.N)), by = .(method, cumbudget_scaled)]

g = ggplot(aes(x = cumbudget_scaled, y = mean, colour = method, fill = method), data = ranks_overall[cumbudget_scaled > 0.10]) +
  geom_line() +
  geom_ribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3) +
  scale_colour_manual(values = values) +
  scale_fill_manual(values = values) +
  labs(x = "Fraction of Budget Used", y = "Mean Rank", colour = "Optimizer", fill = "Optimizer") +
  theme_minimal() +
  theme(legend.position = "bottom", legend.title = element_text(size = rel(0.75)), legend.text = element_text(size = rel(0.75)))
ggsave("plots/anytime_average_rank_mf.png", plot = g, device = "png", width = 6, height = 4)

library(scmamp)  # 0.3.2
best_agg = agg_budget[cumbudget_scaled == 0.25]  # switch to 1 for final
best_agg[, problem := paste0(scenario, "_", instance)]
tmp = - as.matrix(dcast(best_agg, problem ~ method, value.var = "mean")[, -1])
friedmanTest(tmp) # 0.25: chi(6) 69.664, p < 0.001
png("plots/cd_025_mf.png", width = 6, height = 4, units = "in", res = 300, pointsize = 10)
plotCD(tmp, cex = 1)
dev.off()

best_agg = agg_budget[cumbudget_scaled == 1]  # switch to 1 for final
best_agg[, problem := paste0(scenario, "_", instance)]
tmp = - as.matrix(dcast(best_agg, problem ~ method, value.var = "mean")[, -1])
friedmanTest(tmp) # 1: chi(6) 83.957, p < 0.001
png("plots/cd_1_mf.png", width = 6, height = 4, units = "in", res = 300, pointsize = 10)
plotCD(tmp, cex = 1)
dev.off()

