# Time Complexity
# Space Complexity
o(n) space = assigning variables equal to the amount of elements...
o(1) space = constant amount of variables used?

usually between o(n) and o(1) complexity?
# Algorithms
## Sorting
### Quicksort
### Merge Sort
### Timsort
### Heapsort
### Bubble Sort
### Insertion Sort
### Selection Sort
### Tree Sort
### Shellsort
### Bucket Sort
### Radix Sort
### Counting Sort
### Cubesort
## Searching
### Linear Search
for-loop to iterate through all elements; return once element is found; O(n) time

improved version is to search front/back at the same time w/ 2 pointers, and break once pointers meet; should be 2x as fast

alternatives: binary search (O(logn) time), hash table (O(1) time)

[Linear Search - GeeksforGeeks](https://www.geeksforgeeks.org/linear-search/)
### Binary Search
(note: requires array to be sorted to work)

find the middle of the array; compare the element in middle to the target. 

return if middle = target.

else, take subarray of either left or right half and repeat; if no match when theres nothing left to split, not in the array

ex: ascending array values[100]; looking for 32. middle value is values[49]?; greater than target, so compare values[24]?; less than target, so compare values[37?38? need to look into this part]

O(logn) time, since values to search is halved each iteration

`Auxiliary Space: O(1) in case of iterative implementation. In case of recursive implementation, O(Logn) recursion call stack space.`

(how to get middle if array is even?)

[Binary Search - GeeksforGeeks](https://www.geeksforgeeks.org/binary-search/)
## Graph Traversal
### Depth-first Search (DFS)
### Breadth-first Search (BFS)
### Dijkstra
### Topological Sort
### A*
## Unsorted
* minimax?
* manachers?