# [1299. Replace Elements with Greatest Element on Right Side](https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        # Last element should be -1; start with it
        right_max = -1
        # Start at the end, work backwards
        for i in range(len(arr) - 1, -1, -1):
            # Store the rightmost max for the next iteration as either the current max, or the current element
            new_max = max(right_max, arr[i])
            # Set the current element to the rightmost max
            arr[i] = right_max
            # Set the rightmost max to the new max calculated in this iteration
            right_max = new_max
        return arr
```
# [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        max_area = 0
        # Store the histogram columns in a stack of its starting index and height
        stack = []
        
        for i, current_height in enumerate(heights):
            # Set the starting index to the current index
            start_index = i
            # Continuously check if the top of the stack is greater than the current column; 
            # if it is, the max area it can contribute can't get higher due to the lower height column,
            # so calculate its max height and pop it
            while stack and stack[-1][1] > current_height:
                index, height = stack.pop()
                # The max height is either the current max height, or
                # the height of the popped column * the difference of the current index and the popped column's start index
                max_area = max(max_area, height * (i - index))
                # The actual starting index of the current column can be moved back, 
                # considered as the start index of the popped column
                start_index = index
            # Append the current column to the stack
            stack.append((start_index, current_height))
        # If any columns remain at the end, calculate their max heights
        for i, height in stack:
            max_area = max(max_area, height * (len(heights) - i))
        return max_area
```
# [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def numTrees(self, n: int) -> int:
        num_trees = [1] * (n + 1)
        
        # number of trees for 0 and 1 nodes is 1; start at 2
        for nodes in range(2, n + 1):
            total = 0
            for root in range(1, nodes + 1):
                # "nodes" is the highest number, "root" is the currently selected number
                left = root - 1
                right = nodes - root
                # The total number of trees for a given number of nodes is 
                # number of trees on the left side * number of trees on the right side
                total += num_trees[left] * num_trees[right]
            num_trees[nodes] = total
        return num_trees[n]
```
# [129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/)
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
    def sumNumbers(self, root: TreeNode) -> int:
        def dfs(node, num):
            # If the node is null, return 0
            if not node:
                return 0
            # Multiply the given number by 10 to open the least significant digit, then add node's value to it
            num = num * 10 + node.val
            # If leaf node, return the number to prevent returning 0s
            if not node.left and not node.right:
                return num
            # If not leaf node, recurse down the left and right sides to get the full number; 
            # the completed numbers will get added up here in pairs of 2 and be sent up to the root
            return dfs(node.left, num) + dfs(node.right, num)
        return dfs(root, 0)
```
# [148. Sort List](https://leetcode.com/problems/sort-list/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        left = head
        right = self.find_mid(head)
        temp = right.next
        right.next = None
        right = temp
        
        left = self.sortList(left)
        right = self.sortList(right)
        return self.merge_lists(left, right)
    # See Middle of the Linked List for details
    # TODO: why does fast start at head.next and not head?
    def find_mid(self, head):
        fast, slow = head.next, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
    # See Merge Two Sorted Lists for details
    def merge_lists(self, list1, list2):
        dummy = ListNode()
        tail = dummy
        while list1 and list2:
            if list1.val > list2.val:
                tail.next = list2
                list2 = list2.next
            else:
                tail.next = list1
                list1 = list1.next
            tail = tail.next
        if list1:
            tail.next = list1
        if list2:
            tail.next = list2
        return dummy.next
```
# [2. Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode()
        current = dummy
        # Use int to store the carry digit if the addition result is >= 10
        carry = 0
        # the lists may have unequal lengths, so null check before getting their values;
        # it's also possible that both lists are null but a carry remains, so check the carry also
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            # Add both node values and the carry
            val = val1 + val2 + carry
            # If value is >= 10, store the most significant digit in the carry, and mod the value by 10
            carry = val // 10
            val = val % 10
            # Append a new node containing the result to the result linked list
            current.next = ListNode(val)
            
            # Increment nodes
            current = current.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next
```
# [202. Happy Number](https://leetcode.com/problems/happy-number/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def isHappy(self, n: int) -> bool:
        # Use a set to record numbers that have appeared previously; 
        # if the same number appears twice, it will be an infinite loop
        visited = set()
        while n not in visited:
            visited.add(n)
            # Calculate the new value of n and repeat the loop if needed
            n = self.sum_squares(n)
            
            # If n = 1, it'll stay there; return true
            if n == 1:
                return True
    
    # Mod n by 10 to get the least significant digit, then use integer division to decrease n by a factor of 10 until 0
    def sum_squares(self, n):
        result = 0
        while n:
            result += ((n % 10) ** 2)
            n = n // 10
        return result
```
# [25. Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(0, head)
        previous_group_last = dummy
        
        # TODO: understanding not full yet; return
        # Find the kth element, then reverse LL until you approach kth element's next (swap kth, stop here)
        while True:
            # Search ahead to get the relative kth node, if it exists; if not, don't swap and break
            kth_node = self.get_kth_node(previous_group_last, k)
            if not kth_node:
                break
            next_group_first = kth_node.next
            previous_node = kth_node.next
            current_node = previous_group_last.next
            
            # Reverse the k-sized linked list group until you approach the end (next_group_first)
            while current_node != next_group_first:
                next_node = current_node.next
                current_node.next = previous_node
                previous_node = current_node
                current_node = next_node
                
            # After swapping, kth element is at the start, 
            # and the first element is at the end (previous_group_last.next)
            temp = previous_group_last.next
            previous_group_last.next = kth_node
            previous_group_last = temp
            
        return dummy.next
    
    def get_kth_node(self, current_node, k):
        while current_node and k > 0:
            current_node = current_node.next
            k -= 1
        return current_node
```
# [234. Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        # find middle > reverse the latter half > iterate through both and compare values
        
        # Process for finding middle of linked list (see Middle of the Linked List)
        fast, slow = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # Process for reversing linked list (see Reverse Linked List)
        prev_node = None
        while slow:
            next_node = slow.next
            slow.next = prev_node
            prev_node = slow
            slow = next_node
        
        # Process for checking for palindromes; 
        left_node, right_node = head, prev_node
        # Check right_node since it's shorter than left_node
        while right_node:
            if left_node.val != right_node.val:
                return False
            left_node = left_node.next
            right_node = right_node.next
        return True
```
# [88. Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # Length of nums1 is guaranteed to be # of elements in nums1 and nums2
        last = len(nums1) - 1
        
        # Start from the end of nums1, and fill with the greater value between the last elements of nums1 and nums2;
        # since both lists are sorted, this works
        while m > 0 and n > 0:
            if nums1[m - 1] > nums2[n - 1]:
                nums1[last] = nums1[m - 1]
                m -= 1
            else:
                nums1[last] = nums2[n - 1]
                n -= 1
            last -= 1
        
        # If list1 ran out of elements first, place the rest of list2 elements in
        while n > 0:
            nums1[last] = nums2[n - 1]
            n -= 1
            last -= 1
```
# [146. LRU Cache](https://leetcode.com/problems/lru-cache/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Utilize doubly-linked list to track recency
class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = None
        self.next = None
        
class LRUCache:
    # Use dict to store cache values; initialize 2 placeholder nodes for left/right
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        
        self.left, self.right = Node(0, 0), Node(0, 0)
        self.left.next = self.right
        self.right.prev = self.left
    
    # Remove the given node from the linked list
    def remove(self, node):
        next_node = node.next
        prev_node = node.prev
        
        prev_node.next = next_node
        next_node.prev = prev_node
    
    # Insert the given node into the linked list on the right side
    # TODO: why in the middle of the rightmost instead of to the right of the rightmost?
    def insert(self, node):
        prev_node = self.right.prev
        next_node = self.right
        
        prev_node.next = node
        next_node.prev = node
        
        node.next = next_node
        node.prev = prev_node
        return

    def get(self, key: int) -> int:
        # Whenever a key is accessed, remove and re-insert the node that was accessed;
        # Removing a node from the linked list doesn't remove it from the dict, so no issue
        if key in self.cache:
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].val
        return -1
        

    def put(self, key: int, value: int) -> None:
        # If the key is already in the cache, remove it from the linked list to prevent duplicates
        if key in self.cache:
            self.remove(self.cache[key])
        # Create a new node to store in the dict, and connect it to the linked list
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])
        
        # If the cache is over capacity, remove the leftmost node from the linked list 
        # and its value removed from the cache
        if len(self.cache) > self.capacity:
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```
# [203. Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(0, head)
        previous_node, current_node = dummy, head
        
        while current_node:
            # If the current node's value should be removed, redirect the previous node's pointer to the next node;
            # if not, set the previous node to the current node
            if current_node.val == val:
                previous_node.next = current_node.next
            else:
                previous_node = current_node
            # Set current node to the next node
            current_node = current_node.next
        return dummy.next
```
# [120. Triangle](https://leetcode.com/problems/triangle/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # Initialize DP array of size len(triangle) + 1; 
        # extra space to account for base case of 0s row
        dp = [0] * (len(triangle) + 1)
        
        # Iterate through triangle rows backwards
        for row in triangle[::-1]:
            # To calculate the cost of the minimum path at an index, 
            # add the current number at row[i] to the minimum of dp[i] and dp[i+1]
            for i, num in enumerate(row):
                dp[i] = num + min(dp[i], dp[i + 1])
        # The minimum cost of the first row is stored in the first index
        return dp[0]
```
# [47. Permutations II](https://leetcode.com/problems/permutations-ii/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        results = []
        current_permutation = []
        # Fill a dict with the total number counts for every number in nums
        num_counts = { num: 0 for num in nums }
        for num in nums:
            num_counts[num] += 1
            
        def generate_permutations():
            # If the length of the current permutation = length of nums, add it to results and return (go back)
            if len(current_permutation) == len(nums):
                results.append(current_permutation.copy())
                return
            # Loop through all numbers; if they have counts remaining, try combinations with them
            for num in num_counts:
                if num_counts[num] > 0:
                    current_permutation.append(num)
                    num_counts[num] -= 1
                    generate_permutations()
                    num_counts[num] += 1
                    current_permutation.pop()
        
        generate_permutations()
        return results
```
# [876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        fast, slow = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```
# [108. Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
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
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def create_node(start, end):
            # If start is higher than end, there's no elements
            if start > end:
                return None
            # Find the middle and make it the root; if even number, floor the index
            mid = (start + end) // 2
            root = TreeNode(nums[mid])
            # Take the elements to the left and right of the middle to create the left and right children
            root.left = create_node(start, mid - 1)
            root.right = create_node(mid + 1, end)
            return root
        return create_node(0, len(nums) - 1)
```
# [199. Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/)
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
    def rightSideView(self, root: TreeNode) -> List[int]:
        # Solutiotn is the same as Binary Tree Level Order Traversal, but you only care about the last value
        result = []
        queue = collections.deque()
        queue.append(root)
        while queue:
            rightmost_node = None
            queue_length = len(queue)
            for i in range(queue_length):
                node = queue.popleft()
                if node:
                    queue.append(node.left)
                    queue.append(node.right)
                    rightmost_node = node
            if rightmost_node:
                result.append(rightmost_node.val)
        return result
```
# [86. Partition List](https://leetcode.com/problems/partition-list/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # Create two lists; lesser holds values < x, greater holds values >= x
        lesser_dummy, greater_dummy = ListNode(), ListNode()
        lesser_tail, greater_tail = lesser_dummy, greater_dummy
        
        # Iterate through all the nodes, assigning the values to the lesser or greater list
        while head:
            if head.val < x:
                lesser_tail.next = head
                lesser_tail = lesser_tail.next
            else:
                greater_tail.next = head
                greater_tail = greater_tail.next
            head = head.next
        
        # Attach the end of the lesser list to the start of the greater list, and set the end of the greater list to null
        lesser_tail.next = greater_dummy.next
        greater_tail.next = None
        return lesser_dummy.next
```
# [617. Merge Two Binary Trees](https://leetcode.com/problems/merge-two-binary-trees/)
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
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        def sum_nodes(node1, node2):
            if not node1 and not node2:
                return None
            val1 = node1.val if node1 else 0
            val2 = node2.val if node2 else 0
            sum_val = val1 + val2
            left_node = sum_nodes(node1.left if node1 else None, node2.left if node2 else None)
            right_node = sum_nodes(node1.right if node1 else None, node2.right if node2 else None)
            return TreeNode(sum_val, left_node, right_node)
        return sum_nodes(root1, root2)
```
# [735. Asteroid Collision](https://leetcode.com/problems/asteroid-collision/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        # Track right-moving asteroids with a stack
        asteroid_stack = []
        
        for asteroid in asteroids:
            # There will only be a collision if the current asteroid is moving left (< 0) and 
            # there are asteroids moving right (stored in the stack)
            while asteroid_stack and asteroid < 0 and asteroid_stack[-1] > 0:
                result = asteroid_stack[-1] + asteroid
                # If right-moving was larger, set left-moving to 0 so loop ends and it won't get added to the stack
                if result > 0:
                    asteroid = 0
                # If left-moving was larger, pop from the stack and process next one, if it exists
                elif result < 0:
                    asteroid_stack.pop()
                # If they were the same size, set left-moving to 0 and pop right-moving
                else:
                    asteroid = 0
                    asteroid_stack.pop()
            if asteroid:
                asteroid_stack.append(asteroid)
        return asteroid_stack
```
# [543. Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
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
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        # max is either (root) 1 + maxleft + maxright, or 1 + max(maxleft, maxright); if node is null, 0
        diameter = 0
        def longest_path(node):
            nonlocal diameter
            if not node:
                return 0
            left_longest = longest_path(node.left)
            right_longest = longest_path(node.right)
            
            # The longest path (diameter) will either be the longest known diameter, or 
            # the longest path of node's left child and right child
            diameter = max(diameter, left_longest + right_longest)
            return max(left_longest, right_longest) + 1
        longest_path(root)
        return diameter
```
# [337. House Robber III](https://leetcode.com/problems/house-robber-iii/)
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
    def rob(self, root: TreeNode) -> int:
        # 2 choices: don't include root and take the max of root's children, or
        # include root node and take the max of root's children's children (their max without root)
        def max_rob(node):
            # If node is null, only options are 0
            if not node:
                return [0, 0]
            left_max = max_rob(node.left)
            right_max = max_rob(node.right)
            
            with_root = node.val + left_max[1] + right_max[1]
            without_root = max(left_max) + max(right_max)
            
            return [with_root, without_root]
        return max(max_rob(root))
```
# [463. Island Perimeter](https://leetcode.com/problems/island-perimeter/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        
        visited = set()
        def perimeter_dfs(row, col):
            # If a tile is a location that's water or out of bounds, adds a perimeter part to surrounding land
            if row >= rows or col >= cols or row < 0 or col < 0 or grid[row][col] == 0:
                return 1
            # Skip tile if it's not water and has already been visited
            if (row, col) in visited:
                return 0
            visited.add((row, col))
            # Get perimeter by searching surrounding tiles for their perimeters
            perimeter = 0
            perimeter += perimeter_dfs(row + 1, col)
            perimeter += perimeter_dfs(row - 1, col)
            perimeter += perimeter_dfs(row, col + 1)
            perimeter += perimeter_dfs(row, col - 1)
            return perimeter
        
        # Search entire grid to find the first land tile to execute perimeter calculation
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 1:
                    return perimeter_dfs(row, col)
```
# [392. Is Subsequence](https://leetcode.com/problems/is-subsequence/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        # Search both strings with 2 pointers
        i, j = 0, 0
        while i < len(s) and j < len(t):
            # If the substring and longer string match at the character, increment substring pointer
            if s[i] == t[j]:
                i += 1
            # Always increment the longer string that's being searched
            j += 1
        # If i reached the end of the substring, there was a full match; else, there wasn't
        return True if i == len(s) else False
```
# [138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        # Store references to copy nodes with a dict with keys being the original nodes;
        # mapping null to null for edge case of null keys in next/random; 
        # should be able to alternatively do an if null check before assignment
        copy_nodes = { None: None }
        # Loop through once to create all copy nodes, without mapping
        cur = head
        while cur:
            copy_nodes[cur] = Node(cur.val)
            cur = cur.next
        # Loop again to map copy nodes next/random pointers to their respective copies
        cur = head
        while cur:
            copy_node = copy_nodes[cur]
            copy_node.next = copy_nodes[cur.next]
            copy_node.random = copy_nodes[cur.random]
            cur = cur.next
        # Return the copy of the head from the dict
        return copy_nodes[head]
```
# [46. Permutations](https://leetcode.com/problems/permutations/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# This is actually the solution for Permutations II, but the exact same code works; normally, 
# you would just pop numbers from nums and append it to current_permutation, recurse, 
# then push it back onto nums, but that method doesn't work if there's duplicates?
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        results = []
        current_permutation = []
        # Fill a dict with the total number counts for every number in nums
        num_counts = { num: 0 for num in nums }
        for num in nums:
            num_counts[num] += 1
            
        def generate_permutations():
            # If the length of the current permutation = length of nums, add it to results and return (go back)
            if len(current_permutation) == len(nums):
                results.append(current_permutation.copy())
                return
            # Loop through all numbers; if they have counts remaining, try combinations with them
            for num in num_counts:
                if num_counts[num] > 0:
                    current_permutation.append(num)
                    num_counts[num] -= 1
                    generate_permutations()
                    num_counts[num] += 1
                    current_permutation.pop()
        
        generate_permutations()
        return results
```
# [10. Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        i, j = 0, 0
        cache = {}
        def check_match(i, j):
            # If a match result has been previously confirmed for both strings at 2 given indicies, return it
            if (i, j) in cache:
                return cache[(i, j)]
            # If both indices for the string and pattern are out of bounds, it's a match
            if i >= len(s) and j >= len(p):
                return True
            # If the pattern is out of bounds but the string isn't, no match
            if j >= len(p):
                return False
            # Logic for determining a match; make sure the string is still in bounds, 
            # as the string can be shorter than the pattern due to *
            match = i < len(s) and (s[i] == p[j] or p[j] == ".")
            # Case to handle *; check if there's a match without the * (increment j by 2), or
            # check if there's a match with the star (increment i by 1)
            if j + 1 < len(p) and p[j + 1] == "*":
                cache[(i, j)] = check_match(i, j + 2) or (match and check_match(i + 1, j))
                return cache[(i, j)]
            # If there's a character match, check for a match for the next characters in the string and pattern
            if match:
                cache[(i, j)] = check_match(i + 1, j + 1)
                return cache[(i, j)]
            # If there's no match and no star, no match
            else:
                cache[(i, j)] = False
                return cache[(i, j)]
        return check_match(0, 0)
```
# [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        current_combination = []
        result = []
        def create_combinations(num_open, num_closed):
            # If all opens and closes have been used, add the current combination and return (go back)
            if num_open == num_closed == n:
                result.append("".join(current_combination))
                return
            # If opens can still be used: add one, try combinations with it, then remove it
            if num_open < n:
                current_combination.append("(")
                create_combinations(num_open + 1, num_closed)
                current_combination.pop()
            # If closes can still be used: add one, try combinations with it, then remove it
            if num_closed < num_open:
                current_combination.append(")")
                create_combinations(num_open, num_closed + 1)
                current_combination.pop()
        # Run all possible combinations and return the result
        create_combinations(0, 0)
        return result
```
# [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        # Need to hardcode these to create the combinations
        number_to_letters = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz" 
        }
        result = []
        
        def create_combinations(i, current_string):
            # If the current string is as long as the digits string, done this path; 
            # add the string to the result list and return (start going back)
            if len(current_string) == len(digits):
                result.append(current_string)
                return
            # For every possible character at the current digit, try combinations with it 
            # (recurse on next index with current character included in current_string)
            for char in number_to_letters[digits[i]]:
                create_combinations(i+1, current_string + char)
        if digits:
            create_combinations(0, "")
        return result
```
# [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # The DP grid is 1 tile larger than necessary to account for out of bounds cases (out of bounds = 0)
        dp = [[0 for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]
        # Iterate through the grid backwards to b uild up to the main solution
        for i in range(len(text1) - 1, -1, -1):
            for j in range(len(text2) - 1, -1, -1):
                # If both characters are matching, the LCS starting from text1's ith character and
                # text2's jth character is 1 + LCS of the comparison of the next character in both strings 
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                # If not matching, the LCS is the max of either the LCS of the ith character and the j+1th character, or
                # the LCS of the i+1th character and the jth character
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        # Return the result of the first tile
        return dp[0][0]
```
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
# [1. Two Sum](https://leetcode.com/problems/two-sum/)
## Information
**Category:** Easy

**Patterns:** ?

## Question
Given an array of integers `nums` and an integer `target`, return *indices* of the two numbers such that they *add up* to *`target`*.

You may assume that each input would have ***exactly* one solution**, and you may not use the *same* element twice.

You can return the answer in any order.

**Example 1:**
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
```

**Example 2:**
```
Input: nums = [3,2,4], target = 6
Output: [1,2]
```

**Example 3:**
```
Input: nums = [3,3], target = 6
Output: [0,1]
``` 

**Constraints:**
* `2 <= nums.length <= 103`
* `-109 <= nums[i] <= 109`
* `-109 <= target <= 109`
* **Only one valid answer exists.**

## Solutions
* Double `for` loop; return array iterating indices `i` and `j` if `target == (nums[i] + nums[j])`
* For each number in array, insert into hash table (key = number, value = index) and check if hash table contains key `target - nums[i]`; if it does, return `hashtable.get(target - number)` and `i`
## Notes
the goal for two sum is `target = nums[i] + nums[j]`, so you can search for `nums[j]` by applying algebra to the original equation to create `target - nums[i] = nums[j]`. `nums[i]` and `nums[j]` are from the same array, so by storing `target - nums[i]` in a hash table, you can access `nums[j]` in O(1) time, if it exists

## Solution Code
``` py
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Initialize dictionary for storing {complement: index}
        complements = {}
        for i, num in enumerate(nums):
            # Since the goal is nums[i] + nums[j] = target, 
            # if nums[j] = target - nums[i] exists, we're done
            complement = target - num
            if complement in complements:
                return [i, complements.get(complement)]

            # Store the current number/index, in case we run into its complement in the future
            complements.setdefault(num, i)
```
# [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# TODO: commenting not done yet
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        have_chars, need_chars = {}, {}
        # Populate needed_chars with the character counts of t
        for char in t:
            need_chars[char] = need_chars.get(char, 0) + 1
        # Track the number of distinct letter counts that you currently have, 
        # and distinct letter counts that need to be met
        have_count, need_count = 0, len(need_chars)
        # Track the start/end indices of the result substring for returning, and the length of it
        result_substring, result_length = [-1, -1], float("inf")
        left = 0
        for right in range(len(s)):
            char = s[right]
            have_chars[char] = have_chars.get(char, 0) + 1
            
            if char in need_chars and have_chars[char] == need_chars[char]:
                have_count += 1
            # If all the 
            while have_count == need_count:
                current_length = right - left + 1
                if current_length < result_length:
                    result_substring = [left, right]
                    result_length = current_length
                have_chars[s[left]] -= 1
                if s[left] in need_chars and have_chars[s[left]] < need_chars[s[left]]:
                    have_count -= 1
                left += 1
        if result_length == float("inf"):
            return ""
        return s[result_substring[0]:result_substring[1] + 1]
```
# [121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        known_max_profit = 0
        # Use 2 indices to track current_minimum and current_maximum;
        # current is used because indices will be "reset" to the index of 
        # the known lowest value whenever it's discovered
        current_minimum, current_maximum = 0, 0
        
        for i, price in enumerate(prices):
            if price < prices[current_minimum]:
                current_minimum, current_maximum = i, i
            if price > prices[current_maximum]:
                current_maximum = i
                known_max_profit = max(known_max_profit, (prices[current_maximum] - prices[current_minimum]))
                
        return known_max_profit
```
# [33. Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# TODO: Logic not understood fully still; return to this
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            # Integer division to get the middle of the array
            mid = (left + right) // 2
            # If the middle is the target, return it
            if nums[mid] == target:
                return mid
            # Use knowledge of a sorted array to determine if mid is in the bigger or smaller portion of the array;
            # in a sorted array, every element should be <= the leftmost element.
            # If mid is in the lesser half, ...
            if nums[left] <= nums[mid]:
                if target > nums[mid] or target < nums[left]:
                    left = mid + 1
                else:
                    right = mid - 1
            # If mid is in the greater half, ...
            else:
                if target < nums[mid] or target > nums[right]:
                    right = mid - 1
                else:
                    left = mid + 1
        return -1
```
# [198. House Robber](https://leetcode.com/problems/house-robber/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def rob(self, nums: List[int]) -> int:
        previous_selection, two_back_rob = 0, 0
        max_rob = 0
        for current_num in nums:
            # The maximum amount to rob at the current house is either to reset and start again as 
            # current house + 2 houses back, or select none and keep the value of previous house + all previous robs 
            # (any other combination would lead to a conflict)
            max_rob = max(previous_selection, two_back_rob + current_num)
            # The new "2 houses back" for the next loop will be the current previous house 
            # (which includes all previous robs)
            two_back_rob = previous_selection
            # The new previous selection for the next loop will be the currently selected option
            previous_selection = max_rob
        return max_rob
```
# [213. House Robber II](https://leetcode.com/problems/house-robber-ii/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# The question is the same as House Robber 1, but the max of (all but first house) and (all but last house)
# needs to be taken, and a new edge case of only 1 house due to the new restriction
class Solution:
    def rob(self, nums: List[int]) -> int:
        def max_rob(nums):
            one_back_rob, two_back_rob = 0, 0
            max_rob = 0
            for current_num in nums:
                max_rob = max(one_back_rob, two_back_rob + current_num)
                two_back_rob = one_back_rob
                one_back_rob = max_rob  
            return max_rob
        # If there's only one house, it will be missed by the max function call
        # (excluding first and excluding last will be 0 in array size 1)
        if len(nums) == 1:
            return nums[0]
        return max(max_rob(nums[1:]), max_rob(nums[:-1]))
```
# [268. Missing Number](https://leetcode.com/problems/missing-number/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        # Create a set from the list of numbers
        num_set = set(nums)
        # Check all numbers from 0 to n; if the number isn't in the set, it's the missing number
        for i in range(len(nums) + 1):
            if i not in num_set:
                return i
```
# [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
## Information
## Question
## Solutions
scan array; if 1, bfs/dfs visit all other 1s in the area; # of islands = # of bfs/dfs searches. whether you bfs or dfs shouldn't matter, since you need to check every grid tile
union find? dunno what this is yet though
## Notes
## Solution Code
``` py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        # For tracking tiles that have been visited in bfs()
        visited_tiles = set()
        num_islands = 0
        
        def unvisited_land(row, col):
            return True if grid[row][col] == "1" and (row, col) not in visited_tiles else False
        
        def bfs(row_to_visit, col_to_visit):
            tiles_to_visit = collections.deque()
            tile = (row_to_visit, col_to_visit)
            tiles_to_visit.append(tile)
            visited_tiles.add(tile)
            while tiles_to_visit:
                # In BFS, retrieve the oldest tile and add its surrounding tiles to the queue;
                # to DFS, you would retrieve the newest tile instead (tiles_to_visit.pop() to get right side)
                row, col = tiles_to_visit.popleft()
                directions_to_travel = [[0, -1], [0, 1], [1, 0], [-1, 0]]
                for dr, dc in directions_to_travel:
                    next_row, next_col = row + dr, col + dc
                    if next_row in range(rows) and next_col in range(cols) and unvisited_land(next_row, next_col):
                        next_tile = (next_row, next_col)
                        tiles_to_visit.append(next_tile)
                        visited_tiles.add(next_tile)
                
        for row in range(rows):
            for col in range(cols):
                if unvisited_land(row, col):
                    bfs(row, col)
                    num_islands += 1
        return num_islands
```
# [226. Invert Binary Tree]
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
    def invertTree(self, root: TreeNode) -> TreeNode:
        # Base case; if the node is null, return; this means that 
        # the node that called the function has no child at the location.
        # Not having this leads to null references
        if not root:
            return None
        
        # Swapping nodes
        temp = root.left
        root.left = root.right
        root.right = temp
        
        # Loop cases; inverting the child nodes of current node's children
        self.invertTree(root.left)
        self.invertTree(root.right)
        
        return root
```

# [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
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
    def isValidBST(self, root: TreeNode) -> bool:
        def is_valid(node, left, right):
            if not node:
                return True
            if not (left < node.val and right > node.val):
                return False
            return (is_valid(node.left, left, node.val) and is_valid(node.right, node.val, right))
        
        return is_valid(root, float("-inf"), float("inf"))
```
# [56. Merge Intervals](https://leetcode.com/problems/merge-intervals/)
## Information
## Question
## Solutions
* sort array, then loop thru each element: check if current.end > next.start; if true, merge (something like change current.end to next.end, dunno specifics)
## Notes
## Solution Code
``` py
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # Sort the intervals by start, so overlaps can easily be found
        intervals.sort(key = lambda i : i[0])
        # Start with the first interval inside by default, 
        # so it can be checked on first iteration of the for loop
        merged_intervals = [intervals[0]]
        # Remember to start at the second element, since first element is inserted
        for start_interval, end_interval in intervals[1:]:
            # If the next interval's start time is less than the latest interval's end time, merge the intervals 
            # (done by setting the latest interval's end time to the max of both intervals' end times);
            # else, append the next interval to the merged interval list
            if merged_intervals[-1][1] >= start_interval:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end_interval)
            else:
                merged_intervals.append([start_interval, end_interval])
        return merged_intervals
```
# [217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        # Use dictionary to track previous elements
        num_count = {}
        for num in nums:
            # If the current number already exists in the dictionary, it's a duplicate; return true
            # else, add the number to the dictionary
            if num in num_count:
                return True
            num_count[num] = 1
        return False
```
# [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        # Use 2 nodes; fast_node will increment by 2 every iteration, slow_node by 1
        slow_node = head
        fast_node = head
        
        while fast_node:
            slow_node = slow_node.next
            fast_node = fast_node.next
            
            # If fast_node isn't null (not end of list), increment again; 
            # if fast_node == slow_node (fast_node caught up to slow_node), there's a cycle; return true. 
            # If there's no cycle, fast_node will become null and won't cross paths with slow_node
            if fast_node:
                fast_node = fast_node.next
                if fast_node == slow_node:
                    return True
        return False
```
# [252. Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        # Sort list of intervals by start, so overlaps can easily be found
        intervals.sort(key = lambda i : i[0])
        for i in range(len(intervals) - 1):
            # If the current interval's end is greater than the next interval's start,
            # there's an overlap, so can't attend the next meeting; return false
            if intervals[i][1] > intervals[i + 1][0]:
                return False
        return True
```
# 3. Longest Substring Without Repeating Characters
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left = 0
        contained_chars = set()
        result = 0
        for right in range(len(s)):
            # If expanding the window to the right introduces a duplicate, 
            # use while loop to shrink the window from the left (remove the character at left, then increment left)
            while s[right] in contained_chars:
                contained_chars.remove(s[left])
                left += 1
            # Add the new character after the window shrinks; doing this beforehand may cause it to be removed unintentionally
            contained_chars.add(s[right])
            # Remember to add 1 to get the actual window size (ex: left, right = 0, 0 is window size 1)
            result = max(result, (right - left) + 1)
        return result
```
# [53. Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # Since array is guaranteed to be at least size 1, set known max to lowest possible value;
        # it'll be overwritten once the loop starts.
        # If empty arrays were possible, would need an edge case check
        known_max_sum = float('-inf')
        current_sum = 0
        for num in nums:
            # If the current sum is negative, it should be ignored; reset current_sum to do this
            if current_sum < 0:
                current_sum = 0
            current_sum += num
            # The known max will either be itself, or current sum + current number
            known_max_sum = max(known_max_sum, current_sum)
        return known_max_sum
```
# 128. Longest Consecutive Sequence
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # Create a set from the list of numbers to easily iterate over unique numbers
        existing_nums = set(nums)
        longest_sequence = 0
        # For each number, check if the number before it (number - 1) exists; 
        # if it does, it's part of a longer sequence (ignore these); if it doesn't, it's the start of a sequence;
        # repeatedly search for the numbers after it; if they exist, increment counter by 1 until sequence breaks;
        # when sequence breaks, take the max of known longest sequence and current sequence
        for num in existing_nums:
            if (num - 1) not in existing_nums:
                sequence = 0
                while (num + sequence) in existing_nums:
                    sequence += 1
                longest_sequence = max(longest_sequence, sequence)
        return longest_sequence
```
# 11. Container With Most Water
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        max_water = 0
        while left < right:
            max_water = max(max_water, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_water
```
# [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/)
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
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        def validate(s_node, t_node):
            # Base cases: if both node's parents have no children on a given side, return true, or
            # if one node's parent has a child but the other doesn't, it's not matching; return false
            if not s_node and not t_node:
                return True
            if not s_node or not t_node:
                return False
            # If both nodes have the same value (both are equal), check if their children are also equal
            # to determine if the trees are the same
            if s_node.val == t_node.val:
                return (validate(s_node.left, t_node.left) and
                        validate(s_node.right, t_node.right))

        # Base cases: if the end of the main tree's subtrees is reached, no matching subtree exists, or
        # if the current subtree is matching, return true (as it's found)
        if not s:
            return False
        if validate(s, t):
            return True
        # Loop case: if the current main tree's subtree isn't matching, check its left and right subtrees
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
```
# 15. 3Sum
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # Sort the array to make finding combinations faster; consider implementing no sort solutiotn also?
        nums.sort()
        result = []
        
        for i, num in enumerate(nums):
            # Because array is sorted and combinations must be unique, 
            # if the current number is the same as the previous number, it will yield the same results; skip
            if i != 0 and nums[i - 1] == nums[i]:
                continue
            # Use sorted two sum algorithm to find the remaining 2 numbers
            left, right = i + 1, len(nums) - 1
            while left < right:
                candidate = num + nums[left] + nums[right]
                if candidate > 0:
                    right -= 1
                elif candidate < 0:
                    left += 1
                else:
                    result.append([num, nums[left], nums[right]])
                    left += 1
                    # Since array is sorted and combinations must be unique, continuously increment until 
                    # next number isn't the same as the current number
                    while nums[left] == nums[left - 1] and left < right:
                        left += 1
        return result
```
# [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
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
    def maxPathSum(self, root: TreeNode) -> int:
        # Negative numbers are possible, so start from lowest possible value rather than 0
        max_sum = float('-inf')
        def max_gain(node):
            nonlocal max_sum
            # If the parent has no children, can't select what doesn't exist (return 0 to force no selection)
            if not node:
                return 0
            # The max gain of child nodes is either their max gain (if chosen to include) or 0 (if chosen to not include)
            left_max_gain = max(max_gain(node.left), 0)
            right_max_gain = max(max_gain(node.right), 0)
            # Check to see if node + max sum of both children is the known max; if it is, assign it
            # if a new max is found later, it'll be overwritten
            new_path_max_sum = left_max_gain + node.val + right_max_gain
            max_sum = max(max_sum, new_path_max_sum)
            # The max gain that a node can give (while continuing the path) is 
            # the node's value + the max of the left side or right side; 
            # both sides can only be selected if you start a new path
            return node.val + max(left_max_gain, right_max_gain)
        max_gain(root)
        return max_sum
```
# [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # Use defaultdict(list) here instead of {}, so keys that don't exist initialize as empty lists
        result = defaultdict(list)
        
        for string in strs:
            # Initialize array of 0's size 26
            char_count = [0] * 26
            for char in string:
                # characters start at 'a', so char - a can be used as an offset 
                # (a - a = 0, b - a = 1, ...)
                char_count[ord(char) - ord('a')] += 1
            # The string's letter count is used as the dictionary key, 
            # so all strings with the same letter count (anagrams) are grouped
            result[tuple(char_count)].append(string)
        # Return result.values() here for the list of anagram strings;
        # returning result would give a list of the keys (the tuples) instead of the values (the string lists)
        return result.values()
```
# [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# This is the solution that can be expanded to support any character; for the version that's O(1) space 
# and only considers lowercase alphabetical characters, see solution for LC 49 (Group Anagrams), as it's basically the same thing
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        s_length, t_length = len(s), len(t)
        # If both strings are not the same length, they can't be anagrams
        if s_length != t_length:
            return False
        # Use dictionary to store character counts
        char_count = defaultdict(int)
        
        # Have character appearances in the first string increment the count, 
        # and appearances in the second string decrement by the same amount
        for i in range(s_length):
            char_count[s[i]] += 1
            char_count[t[i]] -= 1
        # Iterate the entire dictionary; if a value isn't 0, they're not anagrams
        # (if they were, the increment/decrements would cancel out to 0)
        for key in char_count:
            if char_count[key] != 0:
                return False
        return True
```
# 19. Remove Nth Node From End of List
## Information
## Question
## Solutions
* 2 pass approach; loop once to find length of the list, loop again to remove nth element; O(2L) = O(L) time (L = list length) since 2 consecutive loops, O(1) space
* 1 pass approach; 2 pointers, where one pointer is n iterations ahead of the other; once the front pointer reaches the end, remove valule of the back pointer; O(L) time since 1 loop, O(1) space
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # Need to return the head, so use dummy.next as a reference to head;
        # whenever you need to return the linked list, use dummy.next as a reference to it, 
        # and work with other variables for the algorithm
        dummy = ListNode(0, head)
        # Singly-linked lists can only see ahead of them; need to make sure that 
        # the node you stop on is the node before the one to be removed
        remove_next_node = dummy
        # Use a second node that's n steps ahead of the node to remove;
        # it's set to dummy.next and not dummy because of remove_next_node removing the node after it,
        # not the node that it actually is
        n_ahead_node = dummy.next
        for _ in range(n):
            n_ahead_node = n_ahead_node.next
        # Increment both nodes, searching for the end of the list
        while n_ahead_node:
            remove_next_node = remove_next_node.next
            n_ahead_node = n_ahead_node.next
        # Assign the next node of the node before the node to remove the value of the node to remove's next node
        remove_next_node.next = remove_next_node.next.next
        return dummy.next
```
# [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # Initialize 2 indices at the start and end of the string for comparison
        left, right = 0, len(s) - 1
        
        while left < right:
            # Filtering for alphanumeric characters only
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True
```
# [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # Used for keeping a reference to the first ListNode (dummy.next)
        dummy = ListNode()
        # The value of tail will change to represent the end of thte list; starts at dummy
        tail = dummy
        while l1 and l2:
            if l1.val <  l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        # One list will run out of elements first; when this happens, 
        # check which one it was and add the rest of that list to the tail
        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2
        return dummy.next
```
# 23. Merge k Sorted Lists
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # Edge case for k == 0
        if not lists or len(lists) == 0:
            return None
        # While the number of lists is greater than 1, repeatedly merge the lists in groups of 2
        while len(lists) > 1:
            merged_lists = []
            # Since the merging creates a new array, increment by 2 to the same list being merged twice
            for i in range(0, len(lists), 2):
                list_1 = lists[i]
                # If there is an odd number of lists, the second list will be null
                list_2 = lists[i + 1] if (i + 1) < len(lists) else None
                merged_lists.append(self.merge_lists(list_1, list_2))
            lists = merged_lists
        # At the end, there will only be one list in the array, as everything else was merged together
        return lists[0]
    
    # Use merge two sorted lists algorithm
    def merge_lists(self, l1, l2):
        dummy = ListNode()
        tail = dummy
        while l1 and l2:
            if l1.val <  l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2
        return dummy.next
```
# [54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        output = []
        # Get the start/end ranges for unvisited rows/columns; 
        # these will change as rows/columns are visited
        row_start, row_end = 0, len(matrix)
        col_start, col_end = 0, len(matrix[0])
        
        while row_start < row_end and col_start < col_end:
            # Collect all elements in the topmost row
            for col in range(col_start, col_end):
                output.append(matrix[row_start][col])
            # The entire topmost row has been obtained; increment row_start by 1 
            # to make the second row the new topmost row (to prevent re-visiting)
            row_start += 1
            
            # Collect all elements in the rightmost column
            for row in range(row_start, row_end):
                output.append(matrix[row][col_end - 1])
            # Decrement col_end by 1 to prevent re-visiting
            col_end -= 1
            
            # Check here to ensure the loop condition is still true, 
            # as row_start and col_end values have changed inside the loop
            if not (row_start < row_end and col_start < col_end):
                break
            
            # Collect all elements in the bottommost row, going backwards
            for col in range(col_end - 1, col_start - 1, -1):
                output.append(matrix[row_end - 1][col])
            row_end -= 1
            
            # Collect all elements in the leftmost column, going backwards
            for row in range(row_end - 1, row_start - 1, -1):
                output.append(matrix[row][col_start])
            col_start += 1
        return output
```
# [73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # Use a boolean variable to act as a 0-marker for the columns, since they would overlap and become impossible to differentiate;
        # can pick either row or column, it shouldn't matter
        is_column_zero = False
        
        rows, cols = len(matrix), len(matrix[0])
        for row in range(rows):
            for col in range(cols):
                # If an element is 0, mark the first element of the row/column it's in as 0; this is fine because 
                # those elements should have been checked already
                if matrix[row][col] == 0:
                    matrix[0][col] = 0
                    # TODO: explain this lol
                    if row > 0:
                        matrix[row][0] = 0
                    else:
                        is_column_zero = True
        # Scan the matrix again (except first row/column, special case); if the first element of the row it's part of is 0 or 
        # first element of the column it's part of is 0, set to 0
        for row in range(1, rows):
            for col in range(1, cols):
                if matrix[0][col] == 0 or matrix[row][0] == 0:
                    matrix[row][col] = 0

        # Use the top-left element and boolean variable to individually zero out the first row/column
        if matrix[0][0] == 0:
            for row in range(rows):
                matrix[row][0] = 0
        if is_column_zero:
            for col in range(cols):
                matrix[0][col] = 0
```
# [48. Rotate Image](https://leetcode.com/problems/rotate-image/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row_start, row_end = 0, len(matrix)
        col_start, col_end = 0, len(matrix[0])
        while row_start < row_end and col_start < col_end:
            offset = 0
            # TODO: explain this, and the reason why it's not offset < len(matrix) - 1
            while offset < ((row_end - 1) - row_start):
                # Store top left in temp variable
                temp = matrix[row_start][col_start + offset]
                # Move bottom left to top left
                matrix[row_start][col_start + offset] = matrix[(row_end - 1) - offset][col_start]
                # Move bottom right to bottom left
                matrix[(row_end - 1) - offset][col_start] = matrix[row_end - 1][(col_end - 1) - offset]
                # Move top right to bottom right
                matrix[row_end - 1][(col_end - 1) - offset] = matrix[row_start + offset][(col_end - 1)]
                # Move temp (top left) to top right
                matrix[row_start + offset][col_end - 1] = temp
                # Increment the offset so different elements are rotated on the next round
                offset += 1
            # Shrink the size of the matrix portion being swapped
            row_start += 1
            row_end -= 1
            col_start += 1
            col_end -= 1
```
# [167. Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
## Information
## Question
## Solutions
## Notes
## Solution Code
``` py
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        while left < right:
            ans = numbers[left] + numbers[right]
            if ans == target:
                return [left + 1, right + 1]
            if ans > target:
                right -= 1
            else:
                left += 1
```