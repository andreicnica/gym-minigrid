from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import itertools


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
        train=True,
    ):
        self.agent_start_pos = agent_pos
        self.agent_start_dir = None if agent_dir is None else np.clip(agent_dir, 0, 4)
        self.goal_start_pos = goal_pos
        self.goal_rand_offset = goal_rand_offset
        self.unwrapped.train = train
        self.eval_id = None

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
            goal_pos = [width - 2, height - 2]
            self.put_obj(Goal(), *goal_pos)

        self.unwrapped._crt_goal_pos = goal_pos

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


class EmptyOODEnvV0(MiniGridEnv):
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
        train=True,
    ):
        self.agent_start_pos = agent_pos
        self.agent_start_dir = None if agent_dir is None else np.clip(agent_dir, 0, 4)
        self.goal_start_pos = goal_pos
        self.goal_rand_offset = goal_rand_offset
        self.unwrapped.train = train
        self.unwrapped.eval_id = None

        max_offset = goal_rand_offset + 1
        test_goals = np.array(list(itertools.product(range(-max_offset, max_offset+1),
                                                     range(-max_offset, max_offset+1))))
        test_goals = test_goals[(np.abs(test_goals) > (max_offset-1)).any(axis=1)]
        test_goals[:, 0] += goal_pos[0]
        test_goals[:, 1] += goal_pos[1]

        # Filter out unavailable positions
        fff = np.all((test_goals >= 1) & (test_goals < size), axis=1)
        test_goals = test_goals[fff]
        self.test_goals = test_goals[(test_goals != np.array([1, 1])).any(axis=1)]

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
            train = self.unwrapped.train
            goal = Goal()
            goal_pos = self.goal_start_pos
            rnd_off = self.goal_rand_offset

            if rnd_off is not None:
                new_goal_pos = (1, 1)
                while new_goal_pos == (1, 1):
                    if train:
                        offx = np.random.randint(-rnd_off, rnd_off + 1)
                        offy = np.random.randint(-rnd_off, rnd_off + 1)
                        offx = np.clip(goal_pos[0] + offx, 1, width - 2)
                        offy = np.clip(goal_pos[1] + offy, 1, height - 2)
                        new_goal_pos = (offx, offy)
                    else:
                        eval_id = self.unwrapped.eval_id
                        if eval_id is None:
                            new_goal_pos = self.test_goals[
                                np.random.randint(len(self.test_goals))].tolist()
                        else:
                            new_goal_pos = self.test_goals[eval_id].tolist()

                goal_pos = new_goal_pos

            self.put_obj(goal, *goal_pos)
            goal.init_pos, goal.cur_pos = goal_pos
        else:
            # Place a goal square in the bottom-right corner
            goal_pos = [width - 2, height - 2]
            self.put_obj(Goal(), *goal_pos)

        self.unwrapped._crt_goal_pos = goal_pos

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


class EmptyOODEnvV1(MiniGridEnv):
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
        max_offset=3,
        train=True,
        rand_generator=123
    ):
        self.agent_start_pos = agent_pos
        self.agent_start_dir = None if agent_dir is None else np.clip(agent_dir, 0, 4)
        self.goal_start_pos = goal_pos
        self.goal_rand_offset = goal_rand_offset
        self.unwrapped.train = train
        self.unwrapped.eval_id = None

        goal_pos_pos = np.array(list(itertools.product(range(-max_offset, max_offset+1),
                                                       range(-max_offset, max_offset+1))))
        goal_pos_pos[:, 0] += goal_pos[0]
        goal_pos_pos[:, 1] += goal_pos[1]

        # Filter out unavailable positions
        fff = np.all((goal_pos_pos >= 1) & (goal_pos_pos < size), axis=1)
        goal_pos_pos = goal_pos_pos[fff]
        goal_pos_pos = goal_pos_pos[(goal_pos_pos != np.array([1, 1])).any(axis=1)]

        np.random.RandomState(rand_generator).shuffle(goal_pos_pos)
        gw = goal_rand_offset * 2 + 1
        gw2 = (goal_rand_offset + 1) * 2 + 1
        train_split = (gw**2) / float(gw2**2)

        self.train_split = train_split

        train_cnt = int(train_split * len(goal_pos_pos))
        self.train_goals = goal_pos_pos[:train_cnt]
        self.test_goals = goal_pos_pos[train_cnt:]

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
                if self.unwrapped.train:
                    goal_pos = self.train_goals[np.random.randint(len(self.train_goals))].tolist()
                else:
                    eval_id = self.unwrapped.eval_id

                    if eval_id is None:
                        goal_pos = self.test_goals[np.random.randint(len(self.test_goals))].tolist()
                    else:
                        goal_pos = self.test_goals[eval_id].tolist()

            self.put_obj(goal, *goal_pos)
            goal.init_pos, goal.cur_pos = goal_pos
        else:
            # Place a goal square in the bottom-right corner
            goal_pos = [width - 2, height - 2]
            self.put_obj(Goal(), *goal_pos)

        self.unwrapped._crt_goal_pos = goal_pos

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


register(
    id='MiniGrid-EmptyOOD-v0',
    entry_point='gym_minigrid.envs:EmptyOODEnvV0'
)


register(
    id='MiniGrid-EmptyOOD-v1',
    entry_point='gym_minigrid.envs:EmptyOODEnvV1'
)
