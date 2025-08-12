
def two_sum(nums, target):
    num_dict = {}
    for i, n in enumerate(nums):
        if n in num_dict:
            return [num_dict[n], i]
        else:
            num_dict[target - n] = i

# Test cases
print(two_sum([2, 7, 11, 15], 9))  # Output: [0, 1]
print(two_sum([3, 2, 4], 6))  # Output: [1, 2]
print(two_sum([3, 3], 6))  # Output: [0, 1]
