import random

random.seed(0)

ads = 5
rounds = 10000
true_ctr = [random.random() for _ in range(ads)]

def simulate(algo):
    clicks = [0]*ads
    shown = [0]*ads
    total = 0

    for t in range(1, rounds+1):

        if algo == "egreedy":
            if random.random() < 0.1:
                ad = random.randint(0, ads-1)
            else:
                ad = max(range(ads), key=lambda i: clicks[i]/(shown[i]+1))

        elif algo == "ucb":
            ad = max(range(ads),
                     key=lambda i: clicks[i]/(shown[i]+1) +
                     (2 * (t)**0.5)/(shown[i]+1))

        else:  # Thompson Sampling
            ad = max(range(ads),
                     key=lambda i: random.betavariate(clicks[i]+1,
                                                      shown[i]-clicks[i]+1))

        reward = random.random() < true_ctr[ad]
        shown[ad] += 1
        clicks[ad] += reward
        total += reward

    return total / rounds

print("Epsilon-Greedy CTR:", simulate("egreedy"))
print("UCB CTR:", simulate("ucb"))
print("Thompson Sampling CTR:", simulate("thompson"))
