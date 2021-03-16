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
# [300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# TODO: implement dp binsearch solution?
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # Initialize DP array as 1, the shortest possible value of a subsequence
        dp = [1] * len(nums)
        # Start from last element and work up to first element, as earlier elements depend on the result of later elements
        for i in range(len(nums) - 1, -1, -1):
            # Iterate through all elements that come after nums[i] to try combinations with their longest subsequences;
            for j in range(i + 1, len(nums)):
                # If an element that comes after nums[i] is greater than nums[i], it's not increasing; ignore
                if nums[i] < nums[j]:
                    # The current known longest possible subsequence at a given position is either
                    # itself (if a previous selection yielded a higher result) or 1 + longest subsequence of current selection
                    # (this is why it's done backwards?); the actual longest can't be known until all combinations are tried?
                    dp[i] = max(dp[i], 1 + dp[j])
        return max(dp)
```
# [133. Clone Graph](https://leetcode.com/problems/clone-graph/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        # Track the clones in a dictionary to prevent duplicate clones
        copies = {}
        def clone_dfs(node):
            # If the node has already been cloned, return the clone
            if node in copies:
                return copies[node]
            copy = Node(node.val)
            copies[node] = copy
            # Recursively clone all neighbors and append them to the clone
            for neighbor in node.neighbors:
                copy.neighbors.append(clone_dfs(neighbor))
            return copy
        # clone_dfs() needs to return a clone to function, so can just call it on root node;
        # also edge case for empty list
        return clone_dfs(node) if node else None
```
# [102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
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
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        output = []
        # Use queue for storing bfs visits
        queue = collections.deque()
        queue.append(root)
        while queue:
            # The length of the queue is equal to the nodes on the current level; 
            # needed to make sure that the level only contains nodes of that level, 
            # as additional nodes are added for the next level while processing the current one
            queue_length = len(queue)
            current_level = []
            # Process all nodes at the current level, appending their value to the list and 
            # adding their children to the queue
            for i in range(queue_length):
                node = queue.popleft()
                # If nodes have no children, null nodes will be added to the queue; check for this
                if node:
                    current_level.append(node.val)
                    queue.append(node.left)
                    queue.append(node.right)
            # Need to check if current_level isn't empty so an empty list isn't pushed into the output;
            # this would occur if all nodes in the queue are null
            if current_level:
                output.append(current_level)
        return output
```
# [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
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
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        counter = 0
        stack = []
        node = root
        # Since k is always guaranteed to be <= n, loop should run indefinitely
        while node or stack:
            # Repeatedly add the current node to the stack, then traverse down the left side
            while node:
                stack.append(node)
                node = node.left
            # When there are no more nodes on the left side, process middle node; 
            # increment a counter to indicate the current iteration; if counter == k, 
            # you're on the kth node, so return it 
            node = stack.pop()
            counter += 1
            if counter == k:
                return node.val
            # Traverse down the right side after processing the left and middle
            node = node.right
```
# [207. Course Schedule](https://leetcode.com/problems/course-schedule/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Populate a dictionary of key = course, value = list of prerequisite courses
        courses = defaultdict(list)
        for course, prereq in prerequisites:
            courses[course].append(prereq)
        
        # Track previously visited locations for dfs break condition; 
        # if the same course appears twice, there's a cycle
        visited = set()
        def can_complete_dfs(course):
            # Base cases: If the course has no prerequisites, return true; 
            # if the course has been visited before, there's a cycle; return false;
            if not courses[course]:
                return True
            if course in visited:
                return False
            # Visit the location, then check to see if all of its prerequisites can be completed;
            # the only case for not completing is if there's a cycle, so if there is one, return false
            visited.add(course)
            for prereq in courses[course]:
                if not can_complete_dfs(prereq):
                    return False
            # If all prerequisites can be completed, the course can be completed; 
            # empty the prerequisite list to indicate that
            courses[course] = []
            return True
        # Iterate through all courses and check if they can be completed;
        # looping like this is fine because there's guaranteed to be courses 0 to numCourses
        for course in range(numCourses):
            if not can_complete_dfs(course):
                return False
        return True
```
