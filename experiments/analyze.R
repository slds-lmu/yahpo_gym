library(data.table)
library(mlr3misc)
library(ggplot2)
library(pammtools)
dat = readRDS("results.rds")
dat[, iter := seq_len(.N), by = .(method, scenario, target, instance, repl)]
dat[, cumbudget := cumsum(epoch), by = .(method, scenario, target, instance, repl)]
dat[, incumbent := cummax(val_accuracy), by = .(method, scenario, target, instance, repl)]

get_incumbent_cumbudget = function(incumbent, cumbudget) {
  budgets = seq(0, 7280, by = 52)
  map_dbl(budgets, function(budget) {
    ind = which(cumbudget <= budget)
    if (length(ind) == 0L) {
      0
    } else {
      max(incumbent[ind])
    }
  })
}

dat_budget = dat[, .(incumbent_budget = get_incumbent_cumbudget(incumbent, cumbudget), cumbudget = seq(0, 7280, by = 52)), by = .(method, scenario, target, instance, repl)]

agg_budget = dat_budget[, .(mean = mean(incumbent_budget), se = sd(incumbent_budget) / sqrt(.N)), by = .(cumbudget, method, scenario, target, instance)]

ggplot(aes(x = cumbudget, y = mean, colour = method, fill = method), data = agg_budget) +
  geom_step(lwd = 1) +
  geom_stepribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3) +
  facet_wrap(~ scenario + instance + target, scales = "free")

ggplot(aes(x = cumbudget, y = mean, colour = method, fill = method), data = agg_budget[cumbudget > 6000]) +
  geom_step(lwd = 1) +
  geom_stepribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3) +
  facet_wrap(~ scenario + instance + target, scales = "free")

