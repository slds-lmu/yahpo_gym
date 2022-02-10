library(data.table)
library(ggplot2)
dat = readRDS("results.rds")
dat[, cumbudget := cumsum(epoch), by = .(method, scenario, target, instance, repl)]
dat[, incumbent := cummax(val_accuracy), by = .(method, scenario, target, instance, repl)]

ggplot(aes(x = cumbudget, y = incumbent, colour = method), data = dat) +
  geom_step(lwd = 2) +
  facet_wrap(~ scenario + instance + target)

