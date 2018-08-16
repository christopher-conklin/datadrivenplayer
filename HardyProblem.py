#This is a Monte Carlo simulation of the Hardy golf problem
#with extensions (Python 3)
import matplotlib.pyplot as plt
import numpy 
import random

def hole_result(prob):
    score = 0
    remaining = 4
    while remaining > 0:
        score += 1
        p = random.random()
        #Bad shot, continue
        if p < prob:
            continue
        #normal shot, 1 less remaining
        remaining -= 1
        #Excellent shot, 1 more less remaining
        if p > 1-prob:
            remaining -=1
    return score

#returns 1 if p1 beats p2, 0 if tie, and -1 if p2 beats p1
def hole_compare(p1,p2):
    r1 = hole_result(p1)
    r2 = hole_result(p2)
    if r1 < r2:
        return 1
    elif r1 == r2:
        return 0
    return -1

# returns tuple with fraction of wins and fraction of losses for p1
def compare_average(p1,p2,trials):
    total_wins = 0
    total_losses = 0
    for i in range(trials):
        res = hole_compare(p1,p2)
        if res == 1:
            total_wins += 1
        if res == -1:
            total_losses += 1
    return (total_wins/trials, total_losses/trials)

#now p1 is a tuple
def smart_hole_result(p1):
    if type(p1) is not tuple and type(p1) is not list:
        return hole_result(p1)
    score = 0
    remaining = 4
    while remaining > 0:
        score += 1
        p = random.random()
        #Check remaining, if it is just 1, switch to second entry
        if remaining == 1:
            prob = p1[1]
        else:
            prob = p1[0]
        #Bad shot, continue
        if p < prob:
            continue
        #normal shot, 1 less remaining
        remaining -= 1
        #Excellent shot, 1 more less remaining
        if p > 1-prob:
            remaining -=1
    return score

# returns tuple with fraction of wins and fraction of losses for p1
# now p1 and p2 are tuples with the normal and cautious probabilities
def smart_compare(p1,p2):
    r1 = smart_hole_result(p1)
    r2 = smart_hole_result(p2)
    if r1 < r2:
        return 1
    elif r1 == r2:
        return 0
    return -1

#again tuples are needed for smart performance
def smart_compare_average(p1,p2,trials):
    total_wins = 0
    total_losses = 0
    for i in range(trials):
        res = smart_compare(p1,p2)
        if res == 1:
            total_wins += 1
        if res == -1:
            total_losses += 1
    return (total_wins/trials, total_losses/trials)

#scoring average, can be smart or dumb dependign on whether prob is a tuple or not
def scoring_average(prob, trials):
    total = 0
    for i in range(trials):
        total += smart_hole_result(prob)
    return total/trials

def smart_rounds(prob,trials):
    rounds = []
    for i in range(trials):
        total = 0
        for j in range(18):
            total += smart_hole_result(prob)
        rounds += [total]
    return rounds


#Match play comparision with smart player
win_probs = []
loss_probs = []
prob_vals = numpy.arange(0.0,0.5,0.01)
num_trials = 1000000
for p in prob_vals:
    result = smart_compare_average((p,0),0,num_trials)
    win_probs += [result[0]]
    loss_probs += [result[1]]

plt.plot(prob_vals,win_probs,'g-', label='Winning')
plt.plot(prob_vals,loss_probs,'r--', label='Losing')
plt.xlabel('Probability $p$ of hitting an excellent or bad shot')
plt.ylabel('Probability against a $p=0$ player')
plt.legend(loc=7)
plt.title("Smart play against a player who makes all pars")

plt.show()


'''
#Scoring average for smart player with various p values
med = []
top25 = []
bot25 = []
prob_vals = numpy.arange(0.0,0.5,0.01)
num_trials = 1000000
for p in prob_vals:
    rounds = smart_rounds((p,0),num_trials)
    med += [numpy.median(rounds)]
    top25 += [numpy.percentile(rounds,25)]
    bot25 += [numpy.percentile(rounds,75)]

plt.plot(prob_vals, med, 'k-.', label='Median')
plt.plot(prob_vals, top25, 'g-', label='25th Percentile')
plt.plot(prob_vals, bot25, 'r--', label='75th Percentile')
plt.xlabel('Probability $p$ of hitting an excellent or bad shot')
plt.ylabel('Round Score')
plt.title('Scoring vs Consistency')
plt.legend(loc='best')
plt.savefig('HardyScoring.jpg')
plt.show()

'''
