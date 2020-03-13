from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class EmptyEnvV2(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=16,
        agent_pos=(1, 1),
        agent_dir=None,
        goal_pos=(8, 8),
        goal_rand_offset=0,
    ):
        self.agent_start_pos = agent_pos
        self.agent_start_dir = None if agent_dir is None else np.clip(agent_dir, 0, 4)
        self.goal_start_pos = goal_pos
        self.goal_rand_offset = goal_rand_offset

        super().__init__(
            grid_size=size,
            max_steps=400,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.goal_start_pos is not None:
            goal = Goal()
            goal_pos = self.goal_start_pos
            rnd_off = self.goal_rand_offset
            if rnd_off is not None:
                new_goal_pos = (1, 1)
                while new_goal_pos == (1, 1):
                    offx = np.random.randint(-rnd_off, rnd_off + 1)
                    offy = np.random.randint(-rnd_off, rnd_off + 1)
                    offx = np.clip(goal_pos[0] + offx, 1, width - 2)
                    offy = np.clip(goal_pos[1] + offy, 1, height - 2)
                    new_goal_pos = (offx, offy)

                goal_pos = new_goal_pos

            self.put_obj(goal, *goal_pos)
            goal.init_pos, goal.cur_pos = goal_pos
        else:
            # Place a goal square in the bottom-right corner
            self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            if self.agent_start_dir is None:
                self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self.agent_start_dir is not None:
            self.agent_dir = self.agent_start_dir

        self.mission = "get to the green goal square"


register(
    id='MiniGrid-Empty-v2',
    entry_point='gym_minigrid.envs:EmptyEnvV2'
)
