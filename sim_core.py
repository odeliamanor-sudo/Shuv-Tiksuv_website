import numpy as np
import heapq
import enum
import random
import pandas as pd

# ==========================
# Constants
# ==========================
HOUR = 60
DAYS = 5
TMAX = 24 * DAYS * HOUR
NUM_WEEKS = 50

ARRIVAL_RATE_2H = [
    14.8, 4.1, 7.2, 31.5, 70.6, 60.6,
    89.2, 118.4, 89.8, 61.0, 60.2, 28.9
]

P_WHATSAPP = 0.40
BOT_TIME = 3.0
IVR_TIME = 2.0
P_WA_FAIL_TO_PHONE = 0.20

PATIENCE_MEAN = 15.0
PATIENCE_SD = 2.0

P_TRANSFER_TO_SENIOR = 0.20

BREAK_DUR = 45.0
BREAK_EARLIEST = 1 * HOUR
BREAK_LATEST_BUFFER = 2 * HOUR + BREAK_DUR


# ==========================
# Event Codes
# ==========================
class EventCode(enum.IntEnum):
    ARRIVAL = 1
    END_CLASSIFICATION = 2
    SERVICE_END = 3
    ABANDON = 4
    SHIFT_CHANGE = 5
    BREAK_END = 6


class Event:
    def __init__(self, t, code, obj=None, extra=None, seq=0):
        self.t = float(t)
        self.code = EventCode(code)
        self.obj = obj
        self.extra = extra
        self.seq = seq

    def __lt__(self, other):
        if self.t != other.t:
            return self.t < other.t
        return self.seq < other.seq


class Request:
    def __init__(self, o_id, arrival_time):
        self.o_id = o_id
        self.arrival_time = float(arrival_time)

        self.type = None              # "phone" / "whatsapp"
        self.group = None             # "fault" / "train" / "join" / "disconnect"
        self.required_group = None    # "fault" / "train_join" / "senior"

        self.patience = None
        self.abandon_time = None

        self.is_abandoned = False
        self.in_service = False
        self.stage = "primary"        # "primary" / "escalated"


class Agent:
    def __init__(self, agent_id, group, shift_start, shift_end):
        self.id = agent_id
        self.group = group
        self.shift_start = float(shift_start)
        self.shift_end = float(shift_end)

        self.load_units = 0
        self.on_break = False

        self.break_planned_start = None
        self.break_taken = False

        self.should_leave = False


class SimulationOneWeek:
    """
    Adds snapshots to support animation playback.
    Each snapshot stores queue lengths + active staff + abandonments so far.
    """
    def __init__(self, seed, snapshot_every_min=10,
                 p_whatsapp=P_WHATSAPP, p_wa_fail=P_WA_FAIL_TO_PHONE, p_escalate=P_TRANSFER_TO_SENIOR,
                 arrival_rate_2h=None):
        self.rng = random.Random(seed)

        self.snapshot_every_min = int(snapshot_every_min)
        self.p_whatsapp = float(p_whatsapp)
        self.p_wa_fail = float(p_wa_fail)
        self.p_escalate = float(p_escalate)
        self.arrival_rate_2h = list(arrival_rate_2h) if arrival_rate_2h is not None else list(ARRIVAL_RATE_2H)

        self.Tnow = 0.0
        self.FEL = []
        self.ev_seq = 0

        self.o_id = 0
        self.q_seq = 0

        self.Qfault = []
        self.Qtrain_join = []
        self.Qsenior = []

        self.Active_agents = []
        self.group_index = {"fault": 0, "train_join": 1, "senior": 2}

        self.idle_count_by_group = np.zeros(3, dtype=int)
        self.idle_count_total = 0

        self.Abandon_hour = np.zeros(24)
        self.system_time = {"fault": [], "train": [], "join": [], "disconnect": []}

        self.idletime_by_group = np.zeros(3)
        self.idletime_by_hour = np.zeros(24)

        # Animation snapshots
        self.snapshots = []
        self._next_snapshot_t = 0.0
        self._abandoned_total = 0

    # ---------- helpers ----------
    def _push_event(self, t, code, obj=None, extra=None):
        self.ev_seq += 1
        heapq.heappush(self.FEL, Event(t, code, obj=obj, extra=extra, seq=self.ev_seq))

    def _sample_patience(self):
        raw = self.rng.gauss(PATIENCE_MEAN, PATIENCE_SD)
        return max(0.5, raw)

    def _is_idle(self, ag):
        return (ag.load_units == 0) and (not ag.on_break)

    def _inc_idle(self, ag):
        idx = self.group_index[ag.group]
        self.idle_count_by_group[idx] += 1
        self.idle_count_total += 1

    def _dec_idle(self, ag):
        idx = self.group_index[ag.group]
        self.idle_count_by_group[idx] -= 1
        self.idle_count_total -= 1

    def _queue_push(self, required_group, O):
        self.q_seq += 1
        item = (self.Tnow, self.q_seq, O)
        if required_group == "fault":
            heapq.heappush(self.Qfault, item)
        elif required_group == "train_join":
            heapq.heappush(self.Qtrain_join, item)
        else:
            heapq.heappush(self.Qsenior, item)

    def _queue_pop_valid(self, required_group):
        Q = self.Qfault if required_group == "fault" else (
            self.Qtrain_join if required_group == "train_join" else self.Qsenior
        )
        while Q:
            O = heapq.heappop(Q)[2]
            if O.is_abandoned or O.in_service:
                continue
            if O.abandon_time is not None and self.Tnow >= O.abandon_time:
                O.is_abandoned = True
                continue
            return O
        return None

    def _try_remove_agent(self, ag):
        if (ag in self.Active_agents) and ag.should_leave and self._is_idle(ag):
            self._dec_idle(ag)
            self.Active_agents.remove(ag)
            return True
        return False

    def _sample_service_time(self, O):
        if O.type == "phone":
            if O.group == "fault":
                return self.rng.uniform(4, 6)
            if O.group == "train":
                return self.rng.uniform(5, 8)
            if O.group == "join":
                return self.rng.uniform(4, 12)
            return self.rng.triangular(6, 15, 9)
        else:
            if O.group == "fault":
                return self.rng.triangular(5, 11, 8)
            if O.group == "train":
                return self.rng.triangular(7, 15, 8)
            if O.group == "join":
                return self.rng.uniform(8, 12)
            return self.rng.triangular(10, 19, 13)

    def _try_take_break(self, ag):
        if ag.break_taken or ag.break_planned_start is None or ag.on_break:
            return
        if ag.load_units != 0:
            return
        if self.Tnow + 1e-9 < ag.break_planned_start:
            return
        if self.Tnow + 1e-9 > ag.shift_end:
            return

        if self._is_idle(ag):
            self._dec_idle(ag)

        ag.on_break = True
        ag.break_taken = True
        self._push_event(self.Tnow + BREAK_DUR, EventCode.BREAK_END, obj=ag)

    def _accumulate_idle_time(self, start_t, end_t):
        dt = end_t - start_t
        if dt <= 0:
            return
        self.idletime_by_group += self.idle_count_by_group.astype(float) * dt

        curr = start_t
        while curr < end_t - 1e-9:
            h = int((curr / 60) % 24)
            next_hour = (int(curr / 60) + 1) * 60
            seg_end = min(end_t, next_hour)
            seg_dt = seg_end - curr
            if seg_dt > 0:
                self.idletime_by_hour[h] += self.idle_count_total * seg_dt
            curr = seg_end

    def _snapshot_if_needed(self):
        # Take snapshots at fixed time grid to animate.
        while self._next_snapshot_t <= self.Tnow + 1e-9 and self._next_snapshot_t <= TMAX + 1e-9:
            t = self._next_snapshot_t
            hour = int((t / 60) % 24)
            day = int(t // (24 * 60)) + 1
            self.snapshots.append({
                "t": float(t),
                "day": day,
                "hour": hour,
                "Q_fault": len(self.Qfault),
                "Q_train_join": len(self.Qtrain_join),
                "Q_senior": len(self.Qsenior),
                "agents_fault": sum(1 for a in self.Active_agents if a.group == "fault"),
                "agents_tj": sum(1 for a in self.Active_agents if a.group == "train_join"),
                "agents_sen": sum(1 for a in self.Active_agents if a.group == "senior"),
                "agents_on_break": sum(1 for a in self.Active_agents if a.on_break),
                "abandoned_total": int(self._abandoned_total),
                "idle_total": int(self.idle_count_total),
            })
            self._next_snapshot_t += self.snapshot_every_min

    # ---------- dispatch ----------
    def _dispatch(self):
        progressed = True
        while progressed:
            progressed = False

            # Own queues first
            for ag in self.Active_agents:
                if not self._is_idle(ag) or ag.on_break:
                    continue

                if ag.group == "fault":
                    req = self._queue_pop_valid("fault")
                elif ag.group == "train_join":
                    req = self._queue_pop_valid("train_join")
                else:
                    req = self._queue_pop_valid("senior")

                if req is None:
                    continue

                cost = 2 if req.type == "phone" else 1
                if ag.load_units + cost > 2:
                    continue

                self._dec_idle(ag)
                ag.load_units += cost
                req.in_service = True

                st = self._sample_service_time(req)
                self._push_event(self.Tnow + st, EventCode.SERVICE_END, obj=ag, extra=req)
                progressed = True

            # Seniors help other queues
            for ag in self.Active_agents:
                if ag.group != "senior" or not self._is_idle(ag) or ag.on_break:
                    continue

                req = self._queue_pop_valid("fault")
                if req is None:
                    req = self._queue_pop_valid("train_join")
                if req is None:
                    continue

                cost = 2 if req.type == "phone" else 1
                if ag.load_units + cost > 2:
                    continue

                self._dec_idle(ag)
                ag.load_units += cost
                req.in_service = True

                st = self._sample_service_time(req)
                self._push_event(self.Tnow + st, EventCode.SERVICE_END, obj=ag, extra=req)
                progressed = True

    # ---------- main ----------
    def run(self):
        # reset
        self.Tnow = 0.0
        self.FEL = []
        self.ev_seq = 0

        self.Active_agents = []
        self.idle_count_by_group[:] = 0
        self.idle_count_total = 0

        self.Abandon_hour[:] = 0
        self.system_time = {"fault": [], "train": [], "join": [], "disconnect": []}
        self.idletime_by_group[:] = 0
        self.idletime_by_hour[:] = 0

        self.Qfault = []
        self.Qtrain_join = []
        self.Qsenior = []
        self.q_seq = 0
        self.o_id = 0

        self.snapshots = []
        self._next_snapshot_t = 0.0
        self._abandoned_total = 0

        # shifts
        t = 0.0
        while t <= TMAX + 1e-9:
            self._push_event(t, EventCode.SHIFT_CHANGE)
            t += 8 * HOUR

        # first arrival
        self._push_event(0.0, EventCode.ARRIVAL)

        while self.FEL:
            ev = heapq.heappop(self.FEL)

            # idle accumulate
            self._accumulate_idle_time(self.Tnow, ev.t)

            self.Tnow = ev.t
            if self.Tnow >= TMAX:
                break

            # snapshot grid
            self._snapshot_if_needed()

            if ev.code == EventCode.SHIFT_CHANGE:
                shift_start = self.Tnow
                hour_of_day = int((shift_start // 60) % 24)
                shift_end = shift_start + 8 * HOUR

                if hour_of_day == 0:
                    n_fault, n_tj, n_sen = 1, 1, 1
                    shift_name = "Night"
                elif hour_of_day == 8:
                    n_fault, n_tj, n_sen = 5, 5, 3
                    shift_name = "Morning"
                else:
                    n_fault, n_tj, n_sen = 3, 3, 2
                    shift_name = "Evening"

                for ag in self.Active_agents:
                    ag.should_leave = True

                new_agents = []
                for i in range(n_fault):
                    new_agents.append(Agent(f"{shift_name}_fault_{i}", "fault", shift_start, shift_end))
                for i in range(n_tj):
                    new_agents.append(Agent(f"{shift_name}_trainjoin_{i}", "train_join", shift_start, shift_end))
                for i in range(n_sen):
                    new_agents.append(Agent(f"{shift_name}_senior_{i}", "senior", shift_start, shift_end))

                for ag in new_agents:
                    start_window = shift_start + BREAK_EARLIEST
                    end_window = shift_end - BREAK_LATEST_BUFFER
                    ag.break_planned_start = (
                        self.rng.uniform(start_window, end_window) if end_window > start_window else None
                    )

                for ag in new_agents:
                    self.Active_agents.append(ag)
                    self._inc_idle(ag)

                for ag in list(self.Active_agents):
                    if ag.should_leave:
                        self._try_remove_agent(ag)

                self._dispatch()

            elif ev.code == EventCode.ARRIVAL:
                self.o_id += 1
                O = Request(self.o_id, self.Tnow)

                u_type = self.rng.random()
                O.type = "whatsapp" if u_type < self.p_whatsapp else "phone"

                u_group = self.rng.random()
                if u_group < 0.50:
                    O.group = "fault"
                elif u_group < 0.80:
                    O.group = "train"
                elif u_group < 0.90:
                    O.group = "join"
                else:
                    O.group = "disconnect"

                if O.group == "fault":
                    O.required_group = "fault"
                elif O.group == "disconnect":
                    O.required_group = "senior"
                else:
                    O.required_group = "train_join"

                O.patience = self._sample_patience()

                # classification delay
                if O.type == "whatsapp":
                    if self.rng.random() < self.p_wa_fail:
                        O.type = "phone"
                        delay = BOT_TIME + IVR_TIME
                    else:
                        delay = BOT_TIME
                else:
                    delay = IVR_TIME

                self._push_event(self.Tnow + delay, EventCode.END_CLASSIFICATION, obj=O)

                # next arrival
                hour_idx_2h = int((self.Tnow // 60) % 24) // 2
                lam_per_min = (self.arrival_rate_2h[hour_idx_2h] / 60.0)
                dt = self.rng.expovariate(lam_per_min)
                if self.Tnow + dt < TMAX:
                    self._push_event(self.Tnow + dt, EventCode.ARRIVAL)

            elif ev.code == EventCode.END_CLASSIFICATION:
                O = ev.obj
                self._queue_push(O.required_group, O)
                O.abandon_time = self.Tnow + O.patience
                self._push_event(O.abandon_time, EventCode.ABANDON, obj=O)
                self._dispatch()

            elif ev.code == EventCode.SERVICE_END:
                ag = ev.obj
                O = ev.extra
                if ag not in self.Active_agents:
                    continue

                cost = 2 if O.type == "phone" else 1
                ag.load_units = max(0, ag.load_units - cost)
                O.in_service = False

                escalated_now = False
                if ag.group != "senior" and O.stage == "primary":
                    if self.rng.random() < self.p_escalate:
                        escalated_now = True
                        O.stage = "escalated"
                        O.required_group = "senior"
                        O.patience = self._sample_patience()
                        self._queue_push("senior", O)
                        O.abandon_time = self.Tnow + O.patience
                        self._push_event(O.abandon_time, EventCode.ABANDON, obj=O)

                if not escalated_now:
                    self.system_time[O.group].append(self.Tnow - O.arrival_time)

                if self._is_idle(ag):
                    self._inc_idle(ag)

                self._try_remove_agent(ag)
                if ag in self.Active_agents:
                    self._try_take_break(ag)
                    self._try_remove_agent(ag)

                self._dispatch()

            elif ev.code == EventCode.ABANDON:
                O = ev.obj
                if O.is_abandoned or O.in_service:
                    continue
                O.is_abandoned = True
                h = int((self.Tnow / 60) % 24)
                self.Abandon_hour[h] += 1
                self._abandoned_total += 1

            elif ev.code == EventCode.BREAK_END:
                ag = ev.obj
                if ag not in self.Active_agents:
                    continue
                ag.on_break = False
                if self._is_idle(ag):
                    self._inc_idle(ag)
                self._try_remove_agent(ag)
                self._dispatch()

        # ensure snapshot at end
        self._snapshot_if_needed()
        return self._finalize_outputs()

    def _finalize_outputs(self):
        total_min_fault = (1 + 5 + 3) * 8 * DAYS * 60
        total_min_tj = (1 + 5 + 3) * 8 * DAYS * 60
        total_min_sen = (1 + 3 + 2) * 8 * DAYS * 60
        totals = np.array([total_min_fault, total_min_tj, total_min_sen], dtype=float)

        idle_pct_by_group = np.divide(
            self.idletime_by_group, totals, out=np.zeros(3), where=totals > 0
        ) * 100

        capacity_per_hour = np.zeros(24)
        for h in range(24):
            if 0 <= h < 8:
                agents_total = 3
            elif 8 <= h < 16:
                agents_total = 13
            else:
                agents_total = 8
            capacity_per_hour[h] = agents_total * 60 * DAYS

        idle_pct_by_hour = np.divide(
            self.idletime_by_hour, capacity_per_hour,
            out=np.zeros_like(self.idletime_by_hour),
            where=capacity_per_hour > 0
        ) * 100

        abandon_hour_avg_per_day = self.Abandon_hour / DAYS

        return {
            "Abandon_hour_avg_per_day": abandon_hour_avg_per_day.copy(),
            "system_time": {k: list(v) for k, v in self.system_time.items()},
            "idle_pct_by_group": idle_pct_by_group,
            "idle_pct_by_hour": idle_pct_by_hour,
            "snapshots": list(self.snapshots),
        }


# ==========================
# Public API (used by dashboard.py)
# ==========================
def run_one_week_with_snapshots(seed=2025, snapshot_every_min=10, **params):
    sim = SimulationOneWeek(seed=seed, snapshot_every_min=snapshot_every_min, **params)
    res = sim.run()
    snap_df = pd.DataFrame(res["snapshots"])
    return res, snap_df


def run_replications(num_reps=NUM_WEEKS, seed0=2025, **params):
    Ab_sum = np.zeros(24)
    sys_all = {"fault": [], "train": [], "join": [], "disconnect": []}
    idle_hour_sum = np.zeros(24)
    idle_group_sum = np.zeros(3)

    for i in range(num_reps):
        sim = SimulationOneWeek(seed0 + i, snapshot_every_min=999999, **params)
        res = sim.run()

        Ab_sum += res["Abandon_hour_avg_per_day"]
        for k in sys_all:
            sys_all[k].extend(res["system_time"][k])
        idle_hour_sum += res["idle_pct_by_hour"]
        idle_group_sum += res["idle_pct_by_group"]

    return {
        "num_reps": num_reps,
        "Abandon_hour_avg_per_day": Ab_sum / num_reps,
        "system_time_all": sys_all,
        "idle_hour_avg": idle_hour_sum / num_reps,
        "idle_group_avg": idle_group_sum / num_reps
    }
