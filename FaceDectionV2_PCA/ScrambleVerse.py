import random
# import numpy as np
print('How many players are there?')
playerCount = int(input())
playerGive = [[] for _ in range(playerCount)]
nonLandCount = [[] for _ in range(playerCount)]
for i in range(playerCount):
    print("How many non-land permanent does player ", (i + 1), "have?")
    nonLandCount[i] = int(input())
for i in range(playerCount):
    print("Player ", (i + 1), "will give")
    for j in range(nonLandCount[i]):
        playerToGive = random.randint(1,playerCount)
        print("Item ", j, " to Player ", playerToGive)
    

