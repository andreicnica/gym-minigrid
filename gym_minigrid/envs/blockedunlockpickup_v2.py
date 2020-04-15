import math

from gym_minigrid.minigrid import Ball
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register
from gym_minigrid.minigrid import *


class BlockedUnlockPickupV2(RoomGrid):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, seed=None, full_task=False, with_reward=False):
        room_size = 6
        self.full_task = full_task
        self._randPos = self._rand_pos
        self._carrying = None
        self._with_reward = with_reward

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self._carrying = None

        if self.full_task:
            self._full_task_gen_grid()
        else:
            self._all_initial_state()

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)

        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = self._carrying

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def _all_initial_state(self, ):
        rand_dir = True

        # self.place_agent(0, 0)
        obj, _ = self.add_object(1, 0, kind="box")

        # Make sure the two rooms are directly connected by a locked door
        door, door_pos = self.add_door(0, 0, 0, locked=True)
        # Open door
        door.is_open = self._rand_int(0, 2) == 0
        self.unwrapped.door = door

        # Block the door with a ball
        color = self._rand_color()

        self.unwrapped.blocked_pos = door_pos[0]-1, door_pos[1]

        # 50% blocked door by Ball
        if self._rand_int(0, 2) == 0:
            ball_pos = door_pos[0]-1, door_pos[1]
            self.grid.set(door_pos[0]-1, door_pos[1], Ball(color))
            blocked = True
        else:
            _, ball_pos = self.add_object(self._rand_int(0, 2), 0, 'ball', color)
            blocked = False

        # Place agent
        room_pos = self._rand_int(0, 2)
        room = self.get_room(room_pos, 0)

        not_placed = True
        while not_placed:
            new_pos = room.rand_pos(self)
            ooo = self.grid.get(*new_pos)

            if ooo is not None:
                if (ooo.type == "door" and door.is_open) or ooo.type == "ball":
                    self._carrying = ooo
                    self.grid.set(*new_pos, None)
                else:
                    continue

            not_placed = False

        self.agent_pos = new_pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        # Add a key to unlock the door
        if self._carrying is not None:
            room_pos_key = self._rand_int(0, 2) if door.is_open else room_pos
            self.add_object(room_pos_key, 0, 'key', door.color)
        else:
            # Should place key anywhere - even "over" agent
            # Must also consider the door if it is locked
            if not door.is_open and not blocked and self._rand_int(0, 4) == 0:
                # TODO hardcoded prob for having key
                self._carrying = Key(door.color)
            else:
                room_pos_key = self._rand_int(0, 2) if door.is_open else room_pos
                room = self.get_room(room_pos_key, 0)

                not_placed = True
                while not_placed:
                    new_pos = room.rand_pos(self)
                    ooo = self.grid.get(*new_pos)

                    if ooo is not None or new_pos == self.agent_pos:
                        if new_pos == self.agent_pos:
                            self._carrying = Key(door.color)
                        else:
                            continue
                    else:
                        self.grid.set(*new_pos, Key(door.color))

                    not_placed = False

        self.obj = obj
        self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def _full_task_gen_grid(self, ):
        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0]-1, pos[1], Ball(color))
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self._with_reward:
            if action == self.actions.pickup:
                if self.carrying and self.carrying == self.obj:
                    reward = self._reward()
                    done = True

        return obs, reward, done, info

register(
    id='MiniGrid-BlockedUnlockPickup-v2',
    entry_point='gym_minigrid.envs:BlockedUnlockPickupV2'
)
