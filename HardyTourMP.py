#This is a Monte Carlo simulation of the Hardy golf problem
#with extensions for smart play and a tournament
import matplotlib
#this line is necessary for headless operation
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy 
import random
import multiprocessing as mp
import time
import csv
start = time.time() #for timing the code


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

#players is a list of elements of the form (score, (p1,p2)) corresponding to the score and
#characteristics of each player
def tournament_round(players):
    results = []
    for p in players:
        total = p[0]
        for i in range(18):
            total += smart_hole_result(p[1])
        results += [(total,p[1])]
    return results

def print_results(players):
    players = sorted(players)
    for p in players:
        i = players.index(p)
        if p[0] != players[i-1][0]:
            place = str(i+1)
        #print(f'{place}\t {p}')

def payments(players, cut_but_paid):
    players = sorted(players)
    payout_fraction = (0.18, 0.108, 0.068,
            0.048,0.04,0.036,0.0335,0.031,0.029,0.027,
            0.025,0.023,0.021,0.019,0.018,0.017,0.016,
            0.015,0.014,0.013,0.012,0.0112,0.0104,0.0096,
            0.0088,0.008,0.0077,0.0074,0.0071,0.0068,0.0065,
            0.0062,0.0059,0.00565,0.0054,0.00515,0.0049,
            0.0047,0.0045,0.0043,0.0041,0.0039,0.0037,
            0.0035,0.0033,0.0031,0.0029,0.00274,0.0026,
            0.00252,0.00246,0.0024,0.00236,0.00232,0.0023,
            0.00228,0.00226,0.00224,0.00222,0.0022,0.00218,
            0.00216,0.00214,0.00212,0.0021,0.00208,0.00206,
            0.00204,0.00202,0.002)
    # We determine the places
    payouts = {}
    results = []
    place = 0
    number = 0
    fraction = 0
    j = 0
    for p in players:
        #j = players.index(p) #The issue here is with identical players the index isn't right (grabs the first one)k
        if p[0] != players[j-1][0] and j>0:
            payouts[place]= fraction/number
            place = j
            number = 0
            fraction = 0
        number += 1
        if j<70:
            fraction += payout_fraction[j]
        else:
            fraction += payout_fraction[69]
        if j == len(players)-1:
            payouts[place] = fraction/number

        results += [{'player':p, 'place':place}]
        j+=1
    #Next is the cut but paid category, which will get 0.002 each
    if cut_but_paid != []:
        payouts['cut'] = 0.002
        for p in cut_but_paid:
            results += [{'player':p, 'place':'cut'}]
    #Now we must consider if there needs to be a playoff
    if payouts[0] != 0.18:
        p = 1
        while results[p]['place']==0:
            p+=1
        playoff = results[:p]
        #print(f'playoff between {playoff}')
        #Now we have the players, we take them to sudden death
        playoff = [(0,c) for c in playoff]
        while len(playoff)>1:
            for c in playoff:
                playoff = [(smart_hole_result(c[1]['player'][1]),c[1]) for c in playoff]
            #print(playoff)
            for c in playoff:
                if c[0] > min(playoff, key=lambda x: x[0])[0]:
                    playoff.remove(c)            
        #print(f'Winner is {playoff[0][1]}')
        payouts[1] = (payouts[0]*p-0.18)/(p-1)
        payouts[0] = 0.18
        identified_winner = False
        for itr in range(p):
            if results[itr] != playoff[0][1] or identified_winner:
                results[itr]['place'] = 1
            if results[itr]['place'] == 0:
                identified_winner = True
    #print(payouts)
    for r in results:
        r['payout'] = payouts[r['place']]
        #print(str(r['place'])+'\t'+str(r['player'])+'\t'+str(r['payout']))
        #Change player to just have p value
        r['player'] = r['player'][1]
    return results

        


#Will perform 2 rounds, take the low 70 and ties (unless there are more than 78 who make the cut,
#in which case there is a complicated tiebreaker) and then do 2 more rounds
#The tournament will return a list of dicts with the names 'player', 'place', 'payout'
def tournament(players):
    #first two rounds
    for i in range(2):
        players = tournament_round(players)
       # print(f'ROUND {i+1}')
        #print_results(players)
    #making the cut
    players = sorted(players)
    cut = 69
    while players[cut+1][0] == players[cut][0]:
        cut += 1
    #must go one more so that the cut represents the actual person cut
    cut += 1
    #this is the weird rule that says that more than 78 players requires a tiebreaker, 
    #but the 70 and ties still get paid
    cut_but_paid = []
    cut_not_paid = []
    if cut > 78:
        next_best = 69
        while players[next_best-1][0]==players[next_best][0]:
            next_best -= 1
        cut_not_paid = players[cut:]
        if abs(next_best-69) < abs(cut-69):
            cut_but_paid = players[next_best:cut]
            cut = next_best
            
    else:
        cut_not_paid = players[cut:]
    #print(f'Cut at {players[cut][0]} with player number {cut+1}')
    #print(f'{len(cut_but_paid)} players cut but paid')
    players = players[:cut]
    for i in range(2):
        players = tournament_round(players)
        #print(f'ROUND {i+3}')
        #print_results(players)
    return payments(players,cut_but_paid)+[{'player':p[1],'place':'cut','payout':0} for p in cut_not_paid]
    #print(len(cut_not_paid))

#Class for player prob and result for compiling tournament results
class competitor:
    def __init__(self, pval1,pval2=0, e=0, m=0,c=0,w=0):
        self.p1 = pval1
        self.p2 = pval2
        self.events = e
        self.money = m
        self.cuts_made = c
        self.wins = w
    def entered(self):
        self.events+=1
    def made_cut(self):
        self.cuts_made += 1
    def won(self):
        self.wins += 1
    def add(self, newp):
        self.events += newp.events
        self.money += newp.money
        self.cuts_made += newp.cuts_made
        self.wins += newp.wins
    def pstring(self):
         return str(self.p1)+'\t'+str(self.p2)+'\t'+str(self.events)+'\t'+str(self.money)+'\t'+str(self.cuts_made)+'\t'+str(self.wins)+'\n'
    def print(self):
        print(self.pstring())
    def output(self):
        return [self.p1, self.p2, self.events, self.money, self.cuts_made, self.wins]
        

#Class for list of players
class all_players:
    players = []
    def add_player(self,new_p):
        #we either add the data from this player to a previous one or we just add the new player
        for p in self.players:
            if p.p1 == new_p.p1 and p.p2 == new_p.p2:
                p.add(new_p)
                return
        #getting through this means this p value is not in the list
        self.players += [new_p]
    def add_group(self,group):
        for g in group.players:
            self.add_player(g)
    #event_list is a list of tuples of the form ((p1,p2) to go into the tournament
    def tourney(self,event_list):
        results = tournament([(0,p) for p in event_list])
        for r in results:
            p = competitor(r['player'][0],r['player'][1],1,r['payout'])
            if r['place']!='cut':
                p.made_cut()
                if r['place'] == 0:
                    p.won()
            self.add_player(p)
    def print(self):
        print('p1\tp2\tevents\tmoney\tcuts\twins')
        for p in self.players:
            p.print()
    def write(self):
        FILENAME  = time.strftime("%Y-%m-%d--%H%M%S")+'HardyResults.csv'
        with open(FILENAME, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['p1','p2','events','money','cuts','wins'])
            for p in self.players:
                writer.writerow(p.output())
    def plot_wins(self, show=True):
        res = sorted([(p.p1,p.wins/p.events) for p in self.players])
        if show:
            plt.cla()
        plt.plot(*([x for x in zip(*res)]+['r-.']),label='Wins')
        if show:
            plt.title('Fraction of wins')
            plt.savefig('HardyWins.jpg')
    def plot_cuts(self,show=True):
        res = sorted([(p.p1,p.cuts_made/p.events) for p in self.players])
        if show:
            plt.cla()
        plt.plot(*([x for x in zip(*res)]+['k-']),label='Cuts Made')
        if show:
            plt.title('Fraction of cuts made')
            plt.savefig('HardyCuts.jpg')
    def plot_money(self,show=True,purse=1):
        res = sorted([(p.p1,p.money/p.events*purse) for p in self.players])
        if show:
            plt.cla()
        plt.plot(*([x for x in zip(*res)]+['g--']),label='Earnings')
        if show:
            plt.title('Season earnings (millions of dollars), $%dM total purse' % purse )
            plt.savefig('HardyEarnings.jpg')
    def plot_all(self):
        self.plot_wins(False)
        self.plot_cuts(False)
        self.plot_money(False)
        plt.legend(loc='best')
        plt.savefig('HardyAll.jpg')
    def output_all(self):
        self.write()
        self.plot_wins()
        self.plot_cuts()
        self.plot_money(purse=342)

#This is for running in parallel on raspberry pi
def single_process(num_tourneys):
    #print('Process: %s'%mp.current_process().name)
    pt = all_players()
    for i in range(num_tourneys):
        if i%1000==0:
            print('Running Trial %d Process %s'% (i,mp.current_process().name))
        pt.tourney([(random.choice(numpy.arange(0,0.10001,0.001)),)*2 for x in range(156)])
    return pt

def parallel_run(trials,processes):
    player_totals = all_players()
    if __name__ == '__main__':
        N = trials//processes
        pool = mp.Pool()
        results = [pool.apply_async(single_process, args=(N,)) for i in range(processes)]
        output = [p.get() for p in results]
        for p in results:
            player_totals.add_group(p.get())
        return player_totals


trials = 1000000 #total number of trials
processes = 4
parallel_run(trials,processes).output_all()
#single_process(trials).print()
#player_totals.print()
#player_totals.plot_cuts()
#player_totals.plot_wins()
#player_totals.plot_money(purse=342)


'''
#Printing tournament results
places = []
cut = []
money = []
for player in t_res:
    money += [(player['player'][0],player['payout'])]
    if player['place']=='cut':
        cut += [player['player'][0]]
    else:
        places += [(player['player'][0],player['place'])]

plt.scatter(*zip(*money))
plt.show()
'''
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
print("Time elapsed: ",time.time()-start)
