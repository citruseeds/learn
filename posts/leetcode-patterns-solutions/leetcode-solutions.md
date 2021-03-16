# [100. Same Tree](https://leetcode.com/problems/same-tree/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        def validate(p, q):
            if not p and not q:
                return True
            if p and q and p.val == q.val:
                return validate(p.left, q.left) and validate(p.right, q.right)
            return  False
        return validate(p, q)
```
# [62. Unique Paths](https://leetcode.com/problems/unique-paths/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # Base case: tiles on the topmost/leftmost only have 1 path to reach, since can only move right/down
        # (out of bounds considered as 0)
        dp = [[1] * n for _ in range(m)]
        # Ignore the first row/col since it's already known to be 1
        for col in range(1, m):
            for row in range(1, n):
                # The number of paths to a given tile is equal to the number of paths of the tile above/left of it
                dp[col][row] = dp[col - 1][row] + dp[col][row - 1]
        return dp[m - 1][n - 1]
```
# [322. Coin Change](https://leetcode.com/problems/coin-change/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Initialize DP array of length amount + 1 (calling dp[amount] at end);
        # infinity used as default as dp array stores minimum # of coins for a given amount
        dp = [float("inf")] * (amount + 1)
        # Minimum # of coins to make 0 is 0; base
        dp[0] = 0
        # dp[0] is known, so start at 1; iterate through every coin quantity until amount is reached
        for amt in range(1, amount + 1):
            # Iterate through all possible coins; if valid (>= 0), select either the current known minimum for amount or
            # 1 (accounting for itself) + the currently calculated minimum if it's better
            for coin in coins:
                if amt - coin >= 0:
                    dp[amt] = min(dp[amt], 1 + dp[amt - coin])
        # If no combination was possible for amount, dp[amount] would never be set; return -1 in this case
        return dp[amount] if dp[amount] != float("inf") else -1
```
# [152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_product = max(nums)
        # Store both the minimum and maximum of the current; because of negatives, 
        # the possible maximum could come from either the minimum or maximum
        # 1s are neutral in multiplication, so use this
        prev_min_product, prev_max_product = 1, 1
        for num in nums:
            # Store previous max in temp variable to use it when calculating new previous min
            temp = prev_max_product
            # Choices: multiply current number with the previous product, or ignore it (if it's a 0)
            prev_max_product = max(prev_min_product * num, prev_max_product * num, num)
            prev_min_product = min(prev_min_product * num, temp * num, num)
            max_product = max(max_product, prev_max_product)
        return max_product
```
