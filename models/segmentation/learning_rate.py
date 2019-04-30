def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            return step[0]
    #return steps[-1][0]
    return 1e-7

initial = 1e-3
def step_decay(epoch):
    drop_every = 3
    if epoch % drop_every == 0:
        decay_rate = initial*(1-1.*(epoch/drop_every)/100)**3
        return decay_rate
    return initial*(1-1.*(epoch-drop_every)/drop_every/100)**3

#test = lambda epoch: schedule_steps(epoch, [(2e-4, 2), (3e-4, 4), (1e-4, 6), (5e-6, 8), (2e-4, 10), (1e-4, 40), (2e-5, 65), (1e-5, 70), (1e-6, 80)])
test = lambda epoch: step_decay(epoch)
for i in range(20):
    print(i, test(i))
