import numpy as np


def is_positive_number_power_of_2(n):
    return n & (n - 1) == 0


def __enforce_whole_number__(n, enforce):
    if enforce:
        return max(0, round(n))
    return n


class SimpleDPCounter:
    def __init__(
            self,
            eps: float,
            initial_offset=0,
            enforce_whole_number=False
    ):
        self.eps = eps
        self.initial_offset = initial_offset
        self.enforce_whole_number = enforce_whole_number

        self.curr_time = 0
        self.value = initial_offset

    def count(self, n):
        self.curr_time += 1
        self.value += (n + np.random.laplace(loc=0.0, scale=1.0 / self.eps))
        self.value = __enforce_whole_number__(self.value, self.enforce_whole_number)
        return self.value


class BlockDPCounter:
    def __init__(
            self,
            eps: float,
            initial_offset=0,
            block_size=8,
            enforce_whole_number=False
    ):
        self.eps = eps
        self.initial_offset = initial_offset
        self.block_size = block_size
        self.enforce_whole_number = enforce_whole_number
        self.noise_scale = 2.0 / self.eps

        self.curr_time = 0
        self.value = initial_offset
        self.block_counter = 0
        self.simple_counter = SimpleDPCounter(
            eps=self.eps / 2.0, initial_offset=0, enforce_whole_number=self.enforce_whole_number
        )
        self.last_block_counter_val = 0

    def count(self, n):
        self.curr_time = self.curr_time + 1

        # Block counter always increments by current count
        self.block_counter += n

        # When block changes
        if self.curr_time % self.block_size == 0:
            self.block_counter += np.random.laplace(loc=0.0, scale=self.noise_scale)
            self.block_counter = __enforce_whole_number__(self.block_counter, self.enforce_whole_number)
            self.last_block_counter_val = self.block_counter

            # Simply release the block counter val
            self.value = self.initial_offset + self.last_block_counter_val
            self.simple_counter = SimpleDPCounter(
                eps=self.eps / 2.0, initial_offset=0, enforce_whole_number=self.enforce_whole_number
            )
        else:
            # Within a block simple counter is incremented
            self.value = self.initial_offset + self.last_block_counter_val + self.simple_counter.count(n)
        return self.value


class UnboundedBlockDPCounter:
    def __init__(
            self,
            eps: float,
            initial_offset=0,
            enforce_whole_number=False
    ):
        self.eps = eps
        self.initial_offset = initial_offset
        self.enforce_whole_number = enforce_whole_number

        self.curr_time = 0
        self.value = initial_offset

        self.curr_block_size = 2
        self.curr_block_counter = self.get_new_block_counter(self.curr_block_size)
        self.total_val_until_last_block_counter = 0

    def get_new_block_counter(self, bs):
        return BlockDPCounter(
            eps=self.eps,
            initial_offset=0,
            block_size=bs,
            enforce_whole_number=self.enforce_whole_number
        )

    def count(self, n):
        self.curr_time = self.curr_time + 1
        self.value = self.initial_offset + self.total_val_until_last_block_counter + self.curr_block_counter.count(n)

        # Need to create a new block counter?
        if self.curr_block_counter.curr_time == self.curr_block_size ** 2:
            self.curr_block_size += 1
            self.total_val_until_last_block_counter += self.curr_block_counter.value
            self.curr_block_counter = self.get_new_block_counter(self.curr_block_size)

        return self.value


class LogDPCounter:
    def __init__(
            self,
            eps: float,
            initial_offset=0,
            enforce_whole_number=False

    ):
        self.eps = eps
        self.initial_offset = initial_offset
        self.enforce_whole_number = enforce_whole_number
        self.curr_time = 0
        self.value = self.initial_offset
        self.total_so_far = self.initial_offset

    def count(self, n):
        self.curr_time = self.curr_time + 1
        self.total_so_far += n
        if is_positive_number_power_of_2(self.curr_time):
            noise = np.random.laplace(loc=0.0, scale=1.0 / self.eps)
            self.total_so_far += noise
            self.total_so_far = __enforce_whole_number__(self.total_so_far, self.enforce_whole_number)
            self.value = self.total_so_far

        return self.value


class BoundedBinaryTreeDPCounter:
    def __init__(
            self,
            eps: float,
            max_time,
            initial_offset=0,
            enforce_whole_number=False
    ):
        self.eps = eps
        self.max_time = max_time
        self.initial_offset = initial_offset
        self.enforce_whole_number = enforce_whole_number
        if self.eps == 0:
            self.noise_scale = 0
        else:
            self.noise_scale = np.log2(self.max_time) / self.eps
        self.curr_time = 0
        self.value = self.initial_offset
        self.p_sums = {0: 0}
        self.noisy_p_sums = {0: 0}

    def count(self, n):
        self.curr_time = self.curr_time + 1
        assert self.curr_time <= self.max_time
        binary_representation = bin(self.curr_time)[2:][::-1]
        n_bits = len(binary_representation)
        temp_sum = 0
        for i in range(n_bits):
            if binary_representation[i] == '1':
                self.p_sums[i] = temp_sum + n
                self.noisy_p_sums[i] = self.p_sums[i] + np.random.laplace(loc=0.0, scale=self.noise_scale)
                self.noisy_p_sums[i] = __enforce_whole_number__(self.noisy_p_sums[i], self.enforce_whole_number)
                break
            else:
                temp_sum += self.p_sums[i]
                self.p_sums[i] = 0
                self.noisy_p_sums[i] = 0

        ans = self.initial_offset
        for i in range(n_bits):
            if binary_representation[i] == '1':
                ans += self.noisy_p_sums[i]

        self.value = ans
        return ans


class LogAndBoundedBinaryTreeDPCounter:
    def __init__(
            self,
            eps: float,
            initial_offset=0,
            enforce_whole_number=False
    ):
        self.eps = eps
        self.initial_offset = initial_offset
        self.enforce_whole_number = enforce_whole_number

        self.curr_time = 0
        self.value = initial_offset
        self.log_counter: LogDPCounter = LogDPCounter(
            eps=self.eps / 2.0,
            enforce_whole_number=self.enforce_whole_number
        )
        self.bt_counter: BoundedBinaryTreeDPCounter = None

    def count(self, n):
        self.curr_time = self.curr_time + 1
        if is_positive_number_power_of_2(self.curr_time):
            self.value = self.initial_offset + self.log_counter.count(n)
            self.bt_counter = BoundedBinaryTreeDPCounter(
                eps=self.eps / 2.0,
                max_time=self.curr_time,
                enforce_whole_number=self.enforce_whole_number
            )
        else:
            self.value = self.initial_offset + self.log_counter.count(n) + self.bt_counter.count(n)
        return self.value


class LogAndSimpleDPCounter:
    def __init__(
            self,
            eps: float,
            initial_offset=0,
            enforce_whole_number=False
    ):
        self.eps = eps
        self.initial_offset = initial_offset
        self.enforce_whole_number = enforce_whole_number

        self.curr_time = 0
        self.value = initial_offset
        self.log_counter: LogDPCounter = LogDPCounter(
            eps=self.eps / 2.0,
            enforce_whole_number=self.enforce_whole_number
        )
        self.simple_counter = None

    def count(self, n):
        self.curr_time = self.curr_time + 1
        if is_positive_number_power_of_2(self.curr_time):
            self.value = self.initial_offset + self.log_counter.count(n)
            self.simple_counter = SimpleDPCounter(
                eps=self.eps / 2.0, initial_offset=0, enforce_whole_number=self.enforce_whole_number
            )
        else:
            self.value = self.initial_offset + self.log_counter.count(n) + self.simple_counter.count(n)
        return self.value


# class TwoInSequenceDPCounter:
#     def __init__(
#             self,
#             counter_1_type,
#             counter_2_type,
#             eps: float,
#             transit_time,
#             initial_offset=0,
#             enforce_whole_number=False
#     ):
#         self.counter_1_type = counter_1_type
#         self.counter_2_type = counter_2_type
#
#         self.eps = eps
#         self.transit_time = transit_time
#         self.initial_offset = initial_offset
#         self.enforce_whole_number = enforce_whole_number
#
#         self.counter_1 = get_counter_based_on_type(
#             self.counter_1_type, self.eps, initial_offset=0
#         )
#
#         self.counter_2 = get_counter_based_on_type(
#             self.counter_2_type, self.eps, initial_offset=0
#         )
#
#         self.curr_time = 0
#         self.value = initial_offset
#
#     def count(self, n):
#         self.curr_time = self.curr_time + 1
#
#         if self.curr_time < self.transit_time:
#             self.value = self.initial_offset + self.counter_1.count(n)
#         else:
#             self.value = self.initial_offset + self.counter_1.value + self.counter_2.count(n)
#
#         return self.value
#

class SimpleFollowedByBlockDPCounter:
    def __init__(
            self,
            eps: float,
            initial_offset=0,
            block_size=8,
            transit_time=20,
            enforce_whole_number=False
    ):
        self.eps = eps
        self.initial_offset = initial_offset
        self.block_size = block_size
        self.transit_time = transit_time
        self.enforce_whole_number = enforce_whole_number

        self.counter_1 = SimpleDPCounter(
            eps=self.eps, initial_offset=0, enforce_whole_number=self.enforce_whole_number
        )
        self.counter_2 = BlockDPCounter(
            eps=self.eps, initial_offset=0, block_size=self.block_size, enforce_whole_number=self.enforce_whole_number
        )

        self.curr_time = 0
        self.value = initial_offset

    def count(self, n):
        self.curr_time = self.curr_time + 1

        if self.curr_time <= self.transit_time:
            self.value = self.initial_offset + self.counter_1.count(n)
        else:
            self.value = self.initial_offset + self.counter_1.value + self.counter_2.count(n)

        return self.value


# -------------------------------------- End counter definitions ------------------------

def get_counter_based_on_type(counter_type: str, eps, initial_offset=0, enforce_whole_number=False):
    if counter_type == 'block':
        return UnboundedBlockDPCounter(
            eps=eps, initial_offset=initial_offset,
            enforce_whole_number=enforce_whole_number
        )
    elif counter_type.startswith('block_'):
        return BlockDPCounter(
            eps=eps, initial_offset=initial_offset,
            block_size=int(counter_type[6:]),
            enforce_whole_number=enforce_whole_number
        )
    elif counter_type == 'log_simple':
        return LogAndSimpleDPCounter(
            eps=eps, initial_offset=initial_offset,
            enforce_whole_number=enforce_whole_number
        )
    elif '_follow_' in counter_type:
        transit_time = int(counter_type[len('simple_'):counter_type.find('_follow_')])
        return SimpleFollowedByBlockDPCounter(
            eps=eps, initial_offset=initial_offset,
            transit_time=transit_time,
            block_size=int(counter_type[len('simple_' + str(transit_time) + '_follow_block_'):]),
            enforce_whole_number=enforce_whole_number
        )
    elif counter_type == 'simple':
        return SimpleDPCounter(eps=eps, initial_offset=initial_offset,
                               enforce_whole_number=enforce_whole_number)
    elif counter_type == 'log_bt':
        return LogAndBoundedBinaryTreeDPCounter(eps=eps, initial_offset=initial_offset,
                                                enforce_whole_number=enforce_whole_number)
    elif counter_type.startswith('bt_'):
        return BoundedBinaryTreeDPCounter(eps=eps, max_time=int(counter_type[3:]), initial_offset=initial_offset,
                                          enforce_whole_number=enforce_whole_number)
    else:
        raise Exception(f"Unknown counter type: {counter_type}")
