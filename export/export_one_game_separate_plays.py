
import pandas as pd 
import numpy as np 

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
save_dir="/Users/mwojno01/Desktop/"


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
    

### Get players from first play. assume those are the starters
event=Event(event_dicts[0])
player_ids_dict= event.player_ids_dict
starters = event.moments[0].players
starter_ids = [player.id for player in starters]
starter_names_and_jersey_numbers=[player_ids_dict[id] for id in starter_ids]
team_names_for_starters=[player.team.name for player in starters]

### Reduce to one team
focal_team=team_names_for_starters[0]
opponent_team=[x for x in team_names_for_starters if x!=focal_team][0]
starter_ids_for_focal_team = [starter_ids[i] for i in range(len(starter_ids)) if team_names_for_starters[i]==focal_team]
starter_names_and_jersey_numbers_for_focal_team=[starter_names_and_jersey_numbers[i] for i in range(len(starter_ids)) if team_names_for_starters[i]==focal_team]




### 
# Find events (plays) that are non-overlapping and contain starters
###

verbose=True 

event_retained_dict={} # maps event idx to (event_quarter,start time,end time)
game_clock_at_end_of_previous_play = 721.0 # each quarter has 12 min (720 secs)
current_quarter = 1 

for (i,event_dict) in enumerate(event_dicts):

    event=Event(event_dict)
    if event.moments:  # event.moments could be empty; we exclude those

        ### check if we're in the right quarter to analyze with respect to game clock
        event_quarter=event.moments[0].quarter
        if event_quarter>current_quarter:
            # then break out of event dict for loop, go to next quarter.
            current_quarter+=1
            game_clock_at_end_of_previous_play = 721.0 # each quarter has 12 min (720 secs)

        ### determine if event contains starters
        current_players= event.moments[0].players
        current_player_ids=[player.id for player in current_players]
        event_contains_starters=set(starter_ids_for_focal_team).issubset(current_player_ids)

        ### determine if event doesn't overlap previous event
        event_start_time=event.moments[0].game_clock 
        event_end_time=event.moments[-1].game_clock 
        event_doesnt_overlap_previous=event_start_time < game_clock_at_end_of_previous_play

        ### append if event if warranted
        if event_contains_starters and  event_doesnt_overlap_previous:
            event_retained_dict[i]=(event_quarter, event_start_time, event_end_time)
            game_clock_at_end_of_previous_play = event_end_time

        if verbose:
            print(f"start: {event_start_time:.02f}. end: {event_end_time:.02f}. "
                    f"game clock at end of prev play: {game_clock_at_end_of_previous_play:.02f}. "
                    f"event quarter: {event_quarter}. current quarter {current_quarter} "
                    f"starters: {event_contains_starters}. overlap: {event_doesnt_overlap_previous}" )
                    

### compute some stats:
num_events = len(event_retained_dict)
starters_duration_secs=np.sum([value[1]-value[2] for value in event_retained_dict.values()])

print(f"When starters were in, there were {num_events} plays totaling {starters_duration_secs/60.0:.02f} mins")




### Construct dataset
J = len(starter_ids_for_focal_team)
D =2 # x coord and y coord of location
event_ids_to_retain =list(event_retained_dict.keys())
T_max=np.max([n_moments_per_event[i] for i in event_ids_to_retain])

xs = np.full((n_events,T_max,J,D), np.nan)
for (e, event_id) in enumerate(event_ids_to_retain): 
    event=Event(event_dicts[event_id])
    for (t, moment) in enumerate(event.moments):
        for player in moment.players:
            identity = player.id
            if identity in starter_ids_for_focal_team:
                j = starter_ids_for_focal_team.index(identity)
                x,y=player.x, player.y
                xs[e,t,j,0]=x 
                xs[e,t,j,1]=y

np.save(save_dir+f"basketball_game_{focal_team}_vs_{opponent_team}_all_starter_plays.npy", xs)

