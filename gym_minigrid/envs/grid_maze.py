from collections import deque
import numpy as np
import random

from gym_minigrid.minigrid import *
from gym_minigrid.envs.grid_rooms import GridRooms
from gym_minigrid.register import register

MOVE_VEC = [
    np.array([-1, 0]),
    np.array([0, -1]),
    np.array([1, 0]),
    np.array([0, 1]),
]


def connect_rooms(crt_rooms, goal_room, max_room):
    moves = random.sample(MOVE_VEC, len(MOVE_VEC))

    for im, move in enumerate(moves):
        lx, ly = crt_rooms[-1]
        new_x, new_y = lx + move[0], ly + move[1]
        if 0 <= new_x < max_room and 0 <= new_y < max_room:
            vroom = tuple([new_x, new_y])
            if vroom not in crt_rooms:
                crt_rooms.append(vroom)
                if vroom == goal_room:
                    return True, crt_rooms
                else:
                    reached, new_rooms = connect_rooms(crt_rooms, goal_room, max_room)
                    if reached:
                        return True, new_rooms
                crt_rooms.pop()
    return False, list()


def get_room_neighbour(rix, x, y):
    # Door positions, order is right, down, left, up
    # Indexing is (column, row)
    nx, ny = x, y
    if rix == 0:
        ny += 1
    elif rix == 1:
        nx += 1
    elif rix == 2:
        ny -= 1
    else:
        nx -= 1
    return tuple([nx, ny])


class GridMaze(GridRooms):
    def __init__(self,
                 grid_size=3,
                 goal_center_room=True,
                 close_doors_trials=0.4,
                 **kwargs
                 ):
        num_rows = num_cols = grid_size
        assert num_rows == num_cols, "Square only now"
        self.close_doors_trials = int(num_rows * num_rows * 4 * close_doors_trials)

        super().__init__(num_rows=num_rows, num_cols=num_cols, goal_center_room=goal_center_room,
                         **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        room_size = self.room_size - 1
        max_rooms = self.num_rows
        ag_pos = self.agent_pos
        close_doors_trials = self.close_doors_trials

        ag_start_room = ag_pos // room_size
        goal_room = tuple(self.goal_crt_pos // room_size)

        checked_rooms = [tuple(ag_start_room)]

        reached, rooms = connect_rooms(checked_rooms, goal_room, max_rooms)

        for i in range(close_doors_trials):
            # Pick random room. Try to close random door but not the one connecting ag to goal
            x, y = np.random.randint(0, max_rooms, (2,))
            room = self.room_grid[x][y]
            doors_i = [i for i in range(len(room.door_pos)) if room.door_pos[i] is not None]

            if len(doors_i) == 0:
                continue
            select_door = random.sample(doors_i, 1)[0]
            door_neigh_room = get_room_neighbour(select_door, x, y)

            can_remove_door = True
            # Door positions, order is right, down, left, up
            room_coord = tuple([x, y])
            if room_coord in rooms:
                # Check if door connects to closeby rooms
                rpos = rooms.index(room_coord)
                if rpos > 0 and rooms[rpos - 1] == door_neigh_room:
                    can_remove_door = False
                if rpos < len(rooms) - 1 and rooms[rpos + 1] == door_neigh_room:
                    can_remove_door = False
            if can_remove_door:
                dy, dx = room.door_pos[select_door]
                if dx != ag_pos[0] or dy != ag_pos[1]:
                    # dx, dy = room.door_pos[select_door]
                    self.grid.set(dx, dy, Wall())
                    room.door_pos[select_door] = None


register(
    id="MiniGrid-GridMaze-v0",
    entry_point="gym_minigrid.envs:GridMaze"
)
