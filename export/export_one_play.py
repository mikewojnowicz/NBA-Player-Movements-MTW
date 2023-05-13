
from pprint import pprint 
import pandas as pd 
import numpy as np 

from Game import Game
from Event import Event

"""
Each event (a play?) has different moments (timesteps)
We convert the representation into a tabular one (T,J,D) where D=2
that we can feed into our inference.
"""

###
# Configs
###
path_to_json = "/Users/mwojno01/Downloads/0021500492.json"
save_dir="/Users/mwojno01/Desktop"


###
# Start getting data
###

data_frame = pd.read_json(path_to_json)
event_dicts = data_frame['events'] # a list of dicts, one for each event.
n_events=len(event_dicts)


### Explore number of moments per event
# (So that we can focus on one with lots of moments)
# Turn out index 1 has lots of moments 
n_moments_per_event=[None]*n_events 
for (i,event_dict) in enumerate(event_dicts):
    event=Event(event_dict)
    n_moments_per_event[i]=len(event.moments)

# find an event of interest.
event=Event(event_dicts[1])
    
print("Player ids dictionary is:")
pprint(event.player_ids_dict)

# an alternate,  way to find an event of interest.
# * it's more confusing to me (why are we using a "Game" class to host
#   a single event, and not all of them?).
# * it may have additional useful into.  
#   for instance, it gives the home team and guest team. see below
# * it still has the event as an attribute (game.event) 
event_index = 1
game = Game(path_to_json, event_index)
game.read_json()

print(f"the home team is {game.home_team.name}")
print(f"the guest team is {game.guest_team.name}")

### 
# Grab the x,y coords for each player. Then save
###

T = len(event.moments)

# get the ids for the player.  we assume that an event (play)
# has the same ids for all moments (timesteps)
players = event.moments[0].players
player_ids = [player.id for player in players]
J = len(players)
D =2 # x coord and y coord of location

xs = np.zeros((T,J,D))
for (t, moment) in enumerate(event.moments):
    for player in moment.players:
        identity = player.id
        j = player_ids.index(identity)
        x,y=player.x, player.y
        xs[t,j,0]=x 
        xs[t,j,1]=y

np.save(save_dir+"basketball_play.npy", xs)

### Give info 
teams=[player.team.name for player in players]
info = [(i, player.id, team) for i,(player,team) in enumerate(zip(players,teams))]
pprint(info)


### BELOW THIS IS NOT USED.

moment_id = 0 
moment = game.event.moments[0]

T = len(game.event.moments)

print(f"For this moment, the quarter was {moment.quarter}, "
    f"the game clock was at {moment.game_clock} and the shot clock was {moment.shot_clock}.")

moment.players[0].id
moment.players[0].x
moment.players[0].y

"""
My notes:
In the moments that look like:
[1610612766,202689,24.03735,39.00567,0.0]
we have:
0: time ticks? 
1: player id!  these entries correspond to the player id dicitonary
2: x loc?
3: y loc?
4: ball radius?
"""

# pick first moment
moment_id = 0 
moment = game.event.moments[0]

T = len(game.event.moments)

print(f"For this moment, the quarter was {moment.quarter}, "
    f"the game clock was at {moment.game_clock} and the shot clock was {moment.shot_clock}.")


moment.players[0].id
moment.players[0].x
moment.players[0].y
