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

methods = unique(dat_budget$method)
ranks = map_dtr(unique(dat_budget$scenario), function(scenario_) {
  map_dtr(unique(dat_budget$target), function(target_) {
    map_dtr(unique(dat_budget$instance), function(instance_) {
      map_dtr(unique(dat_budget$repl), function(repl_) {
        map_dtr(unique(dat_budget$cumbudget), function(cumbudget_) {
          res = dat_budget[scenario == scenario_ & target == target_ & instance == instance_ & cumbudget == cumbudget_ & repl == repl_]
          if (nrow(res) == 0L) {
            return(data.table())
          }
          setorderv(res, "incumbent_budget", -1)
          data.table(rank = match(methods, res$method), method = methods, scenario = scenario_, target = target_, instance = instance_, cumbudget = cumbudget_, repl = repl_)
        })
      })
    })
  })
})

ranks_agg = ranks[, .(mean = mean(rank), se = sd(rank) / sqrt(.N)), by = .(method, scenario, target, instance, cumbudget)]

ggplot(aes(x = cumbudget, y = mean, colour = method, fill = method), data = ranks_agg) +
  geom_line(lwd = 1) +
  geom_ribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3) +
  facet_wrap(~ scenario + instance + target, scales = "free")

ranks_total = ranks_agg[, .(mean = mean(mean), se = sd(mean) / sqrt(.N)), by = .(method, scenario, target, cumbudget)]

ggplot(aes(x = cumbudget, y = mean, colour = method, fill = method), data = ranks_total) +
  geom_line(lwd = 1) +
  geom_ribbon(aes(min = mean - se, max = mean + se), colour = NA, alpha = 0.3)


