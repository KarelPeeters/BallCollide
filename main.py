import enum
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot, collections, animation


class EventType(enum.Enum):
    WALL_X = enum.auto()
    WALL_Y = enum.auto()
    OTHER = enum.auto()


@dataclass
class Event:
    t: float
    type: EventType
    i: int
    j: int


class State:
    def __init__(self, field_size: float, ball_radius: float, x: np.array, v: np.array):
        assert ball_radius > 0
        self.ball_radius = ball_radius
        self.field_size = field_size

        self.ball_count = x.shape[0]
        assert x.shape == (self.ball_count, 2)
        assert v.shape == (self.ball_count, 2)

        self.x = x
        self.v = v

    def copy(self):
        return State(
            field_size=self.field_size,
            ball_radius=self.ball_radius,
            x=self.x.copy(),
            v=self.v.copy(),
        )

    def advance_t(self, t: float):
        self.x += t * self.v

    def next_event(self) -> Optional[Event]:
        e1 = self.next_collision_wall()
        e2 = self.next_collision_ball()

        if e1.t <= e2.t:
            return e1
        if e2.t <= e1.t:
            return e2

        # both times are nan
        return None

    def apply_event(self, event: Event):
        bias = 0.000001
        self.advance_t(event.t - bias)

        while True:
            if event.type == EventType.WALL_X:
                # flip x velocity
                self.v[event.i, 0] *= -1
                break
            if event.type == EventType.WALL_Y:
                # flip y velocity
                self.v[event.i, 1] *= -1
                break

            if event.type == EventType.OTHER:
                x1 = self.x[event.i, :]
                x2 = self.x[event.j, :]
                v1 = self.v[event.i, :].copy()
                v2 = self.v[event.j, :].copy()

                d_squared = np.linalg.norm(x1 - x2) ** 2

                self.v[event.i, :] -= np.dot(v1 - v2, x1 - x2) * (x1 - x2) / d_squared
                self.v[event.j, :] -= np.dot(v2 - v1, x2 - x1) * (x2 - x1) / d_squared
                break

            assert False, f"expected event type in {event.type}"

        self.advance_t(bias)

    def next_collision_wall(self) -> Event:
        # find intersect times, first bottom/left then top/right
        t1 = (self.ball_radius - self.x) / self.v
        t2 = (self.field_size - self.ball_radius - self.x) / self.v

        # mask out negative times and nans
        t1[~(t1 >= 0)] = np.inf
        t2[~(t2 >= 0)] = np.inf

        t = np.minimum(t1, t2)
        i, j = np.unravel_index(np.argmin(t), shape=t.shape)
        return Event(
            t=t[i, j],
            type=EventType.WALL_X if j == 0 else EventType.WALL_Y,
            i=i, j=-1
        )

    def next_collision_ball(self) -> Event:
        # maybe there's a way to speed this up, right now we're also computing symmetric and self collisions

        # setup quadratic equation
        dx = self.x[:, None, :] - self.x[None, :, :]
        dv = self.v[:, None, :] - self.v[None, :, :]

        a = np.sum(dv * dv, axis=2)
        b = 2 * np.sum(dx * dv, axis=2)
        c = np.sum(dx * dx, axis=2) - 4 * self.ball_radius ** 2

        # solve the equation
        d = b ** 2 - 4 * a * c
        d_sqrt = np.sqrt(d)
        t1 = (-b + d_sqrt) / (2 * a)
        t2 = (-b - d_sqrt) / (2 * a)

        # mask out negative times and nans
        t1[~(t1 >= 0)] = np.inf
        t2[~(t2 >= 0)] = np.inf

        # find the first time and the corresponding balls
        t = np.minimum(t1, t2)
        i, j = np.unravel_index(np.argmin(t), shape=t.shape)
        return Event(
            t=t[i, j],
            type=EventType.OTHER,
            i=i, j=j
        )


def generate_random_state(n: int, field_size: float, ball_radius: float, speed_sigma: float) -> State:
    x = np.empty((n, 2), dtype=float)

    for i in range(n):
        print(f"Random state progress: {i}/{n}")
        attempts = 0
        while True:
            attempts += 1
            if attempts > 100:
                raise Exception(f"Random init {i + 1}/{n} took >100 attempts")

            new_pos = np.random.uniform(1.5 * ball_radius, field_size - 1.5 * ball_radius, size=2)
            dist = np.linalg.norm(new_pos - x[:i, :], axis=1)

            if np.all(dist > 2.5 * ball_radius):
                x[i, :] = new_pos
                break

    v = np.random.normal(0, speed_sigma, size=(n, 2))

    return State(
        field_size=field_size,
        ball_radius=ball_radius,
        x=x, v=v
    )


def generate_animation_data(state: State, total_time: float, fps: float) -> [State]:
    result = []

    time_left = total_time
    skip_time = 0

    while True:
        print(f"Simulation progress: {1 - time_left / total_time:.4f}")

        # figure out the next event
        event = state.next_event()
        print(event)
        if event is None:
            event_t = np.inf
        else:
            event_t = event.t

        # generate the frames from before this event if any
        if event_t > skip_time:
            end_time = min(time_left, event_t)
            for frame_time in np.arange(skip_time, end_time, 1 / fps):
                copy = state.copy()
                copy.advance_t(frame_time)
                result.append(copy)
            skip_time = frame_time + 1 / fps - end_time
        else:
            skip_time -= event_t

        # update the state if we'll use it
        if event is None or event.t > time_left:
            break
        state.apply_event(event)
        time_left -= event.t

    return result


def plot_animation(states: [State], fps: float):
    fig = pyplot.figure()
    ax = pyplot.gca()
    ax.set_aspect('equal', adjustable='box')
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    def frame(state_index):
        print(f"rendering: {state_index}/{len(states)}")
        ax.clear()
        state = states[state_index]

        circles = [
            pyplot.Circle(state.x[i, :], radius=state.ball_radius)
            for i in range(state.ball_count)
        ]
        ax.add_collection(collections.PatchCollection(circles))

    ani = animation.FuncAnimation(fig, func=frame, frames=len(states), interval=int(1000 / fps))

    print("saving animation")
    os.makedirs("output", exist_ok=True)
    ani.save("output/test.mp4")
    pyplot.savefig("output/test.png")

    pyplot.close()


def main():
    np.random.seed(54657631)
    state = generate_random_state(n=200, field_size=1, ball_radius=0.01, speed_sigma=0.3)

    fps = 60
    states = generate_animation_data(state, 1, fps)
    plot_animation(states, fps)


if __name__ == '__main__':
    main()
