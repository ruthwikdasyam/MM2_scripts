Mobile Manipulator

What do we want the robot to do. Example queries.

Where is my cup, i couldnt find it.
Are there any pens in this room.
im hungry, i need a chocolate


### Low level actions

- go to a particular person or places - travels to ruthwik, zahir, amisha, kasra
- go to a particular point - travels to given pos
- go to particular object - travel near that object
- captures image - returns caption 
- manipulator can pickup and place objects
- It can just wait and observe without any navigation or manipualation
- It can ask for help/clarifications needed at any point



### Abstract

A robot that is programmed to coexist with humans need to behave and act closely to humans.
To achieve this, robot demonstrate behaviors to remember the events, alayze environments, communicate with humans and thus, plans tasks accordingly.  Robot continuously plans for the tasks based on its memory, observation and user inputs. This is achieved by maintaining high level and low level planning by LLM.



---

It should move only when it needs to move, if it can answer from log, dont move

Prompts - ask GPT
1. How many sticky notes are there in the white board
2. is there any monitor around here ?