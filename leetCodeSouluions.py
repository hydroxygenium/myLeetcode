#2nd problem
#status: solved
#
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# class Solution:
#     def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
#         tempSum  = 0
#         # will return temp.next
#         temp = ListNode(0)
        
#         # this will be used to traverse the linkedList and keep adding the sums from L1 and L2
#         curr = temp
        
#         #carry over to be added
#         carry = 0
        
        
#         while l1 or l2 or carry !=0:
#             val1 = l1.val if l1 else 0
#             val2 = l2.val if l2 else 0
            
#             carry, out = divmod(val1+val2+carry,10)
            
#             curr.next  = ListNode(out)
#             curr = curr.next
            
#             l1 = (l1.next if l1 else None)
#             l2 = (l2.next if l2 else None)
        
#         return temp.next


#3rd problem
#status: solved
# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         n=len(s)
#         if not n:
#             return 0
#         if n == 1:
#             return 1
#         res=1
#         visited = [0]*256
#         l,h=0,1
#         visited[ord(s[l])] = 1
#         while h < n:
#             if visited[ord(s[h])] == 1:
#                 while l < h and visited[ord(s[h])] == 1:
#                     visited[ord(s[l])] = 0
#                     l += 1
#             visited[ord(s[h])] = 1
#             if l == h:
#                 h += 1
#             else:
#                 res = max(res,(h-l)+1)
#                 h += 1
#         return res

#4th problem
#status: solved
# class Solution:
#     def findMedianSortedArrays(self, nums1: list, nums2: list):
#         self.nums1 = nums1
#         self.nums2 = nums2
#         nums = nums1+nums2
#         nums.sort()
#         x = len(a)/2
#         if x%2==0:
#             return((a[x-1]+a[x])/2)
#         else:
#             return (a[x])


#6th problem
#status: solved
# class Solution:
#     def convert(self, s: str, numRows: int) -> str:
#         indexes_order = list(range(numRows))
#         indexes_order.extend(list(range(numRows-2, 0, -1)))
#         string_rows = ['']*numRows
#         for index, ch in enumerate(s):
#             pos = index % len(indexes_order)
#             string_rows[indexes_order[pos]] += ch
#         return ''.join(string_rows)


#7th problem
#status: solved
# import re
# class Solution:
#     def myAtoi(self, s: str) -> int:
#         i=0
#         s=s.lstrip() # remove leading spaces
#         sign=''
#         if not s:
#             return 0
#         if(s[0] == '-' or s[0]== '+'):
#                 sign = s[0]
#                 s = s[1:]
#         if re.search("^[0-9]",s):
#                 x=re.search("[^0-9]|$",s)
#                 i=int(s[:x.start()])
#                 i = i*(-1) if (sign =='-') else i
#                 i = pow(2,31)-1 if i>pow(2,31)-1 else i
#                 i = pow(-2,31) if i<pow(-2,31) else i
#         return i


#10th problem
#status: solved
# import re
# class Solution:
#     def isMatch(self, s: str, p: str) -> bool:
#         return re.fullmatch(p, s)


#11th problem
#status: solved
'''class Solution:
    def maxArea(self, height: list[int]) -> int:
        #We cannot sort the list.
        L, R, max_sum, h  = 0, len(height)-1, 0, 0
        while L<R:
            ####Two pointers at the front and last.
            area = R-L
            if height[R] >= height[L]:
                h = height[L]
                ###left pointer is smaller than right pointer,so we move left pointer.
                L+=1
            else:
                h = height[R]
                ###right pointer is smaller than left pointer,so we move right pointer.
                R-=1
            #Update the max_sum area.
            if h * area > max_sum:
                max_sum = h * area
        return max_sum'''


#12th problem
#status: solved
# class Solution:
#     def intToRoman(self, num: int) -> str:
#         val = [
#             1000, 900, 500, 400,
#             100, 90, 50, 40,
#             10, 9, 5, 4,
#             1
#             ]
#         syb = [
#             "M", "CM", "D", "CD",
#             "C", "XC", "L", "XL",
#             "X", "IX", "V", "IV",
#             "I"
#             ]
#         roman_num = ''
#         i = 0
#         while  num > 0:
#             for _ in range(num // val[i]):
#                 roman_num += syb[i]
#                 num -= val[i]
#             i += 1
#         return roman_num


#13th problem
#status: solved
# class Solution:
#     def romanToInt(self, s: str) -> int:
#         values = {"I": 1,
#                   "V": 5,
#                   "X": 10,
#                   "L": 50,
#                   "C": 100,
#                   "D": 500,
#                   "M": 1000,}
#         num = 0
 
#         for i in range(len(s)-1):
#             if values[s[i]] >= values[s[i+1]]:
#                 num += values[s[i]]
#             if values[s[i]] < values[s[i+1]]:
#                 num -= values[s[i]]
#         num += values[s[-1]]

#         return num


#14th problem
#status: solved
# class Solution:
#     def longestCommonPrefix(self, strs: list[str]) -> str:
#         if len(strs) == 0:
#             return ""
#         if len(strs) == 1:
#             return strs[0]
#         # set prefix as the first element in the list
#         prefix = strs[0]
#         prefix_length = len(prefix)
#         # iterate and modify the prefix as per the element in the list
#         for i in strs[1:]:
#             while prefix!=i[0:prefix_length]:
#                 prefix = prefix[0:prefix_length-1]
#                 prefix_length = prefix_length-1
                
#                 if len(prefix) == 0:
#                     return ""
#         return prefix


#15th problem
#status: solved but slow
# from itertools import permutations

# def allComb(string, lenght: int):
#     '''
#     nothing
#     '''
#     outlst = []
#     for comb in permutations(string, lenght):
#         outlst.append(comb)
#     outlst = list(map(list, outlst))

#     return sorted(outlst)


# class Solution:
# 	def threeSum(self, nums: list[int]) -> list[list[int]]:
# 		triplets = allComb(nums, 3)
# 		validTriplets = [i for i in triplets if sum(i)==0]

# 		for j in range (0, len(validTriplets)):
# 			validTriplets[j] = sorted(validTriplets[j])

# 		validOut = []
# 		for i in validTriplets:
# 			if i not in validOut:
# 				validOut.append(i)

# 		return validOut
# #alternative and fast solution
# class Solution:
#     def threeSum(self, nums: list[int]) -> list[list[int]]:
#         if len(nums) < 3:
#             return []
#         elif len(nums) == 3 and sum(nums) == 0:
#             return [nums]

#         zeros = 0
#         negatives = {} # {value => repeat_flag} 
#         positives = {} # {value => repeat_flag}

#         for x in nums:
#             if x < 0:
#                 negatives[x] = 0 if x not in negatives else 1
#             elif x > 0:
#                 positives[x] = 0 if x not in positives else 1
#             else:
#                 zeros += 1

#         negatives = dict(sorted(negatives.items(), reverse=True))
#         positives = dict(sorted(positives.items()))
        
#         result_for_zeros = []

#         if zeros > 0:
#             if zeros >= 3:
#                 result_for_zeros.append([0, 0, 0])
                
#             if len(negatives) < len(positives):
#                 for negative in negatives:
#                     if -negative in positives:
#                         result_for_zeros.append([0, negative, -negative])
#             else:
#                 for positive in positives:
#                     if -positive in negatives:
#                         result_for_zeros.append([0, -positive, positive])
                        
#         if len(negatives) == 0 or len(positives) == 0:
#             return result_for_zeros           

#         # search for positive (a_list) + negative + negative (bc_list) = 0
#         # search for negative (a_list) + positive + positive (bc_list) = 0
#         return result_for_zeros \
#             + self.find(negatives, positives, True) \
#             + self.find(positives, negatives, False) 

#     def find(self, a_list, bc_list, is_negative):
#         result = []
 
#         for a in a_list:
#             processed = set()
            
#             for b in bc_list:
#                 if b in processed:
#                     continue

#                 if is_negative:
#                     if b >= -a:
#                         break
#                 else:
#                     if -b >= a:
#                         break

#                 c = -(a + b)

#                 if c in bc_list:
#                     if c == b and bc_list[c] == 0:
#                         continue

#                     processed.add(c)
#                     result.append([a, b, c])

#         return result


#16th problem
#status: solved
# class Solution:
#     def threeSumClosest(self, nums: list[int], target: int) -> int:
#         gap = 0
#         sign = 1
#         d = {}
#         for i in range(len(nums)):
#             d[nums[i]] = i
            
#         while gap < float('inf'):
#             target += sign * gap
#             sign *= -1
#             gap += 1
            
#             for i in range(len(nums)):
#                 sectarget = target - nums[i]
#                 for j in range(i+1, len(nums)):
#                     need = sectarget - nums[j]
#                     if need in d and d[need] != i and d[need] != j:
#                         return target


#17th problem
#status: solved
# class Solution:
#     def letterCombinations(self, digits: str) -> List[str]:
#         if digits == "": return []
        
#         helper = ["abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
# 		# So we don't neet to covert string to interger every time.
# 		# Optimize: helper[int(digits[i]) - 2] --> helper[dic[digits[i]]]
#         dic = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9}
        
#         res_q = collections.deque([""])
        
# 		# pop out every elements in queue, add the character at the end of each string,
# 		# then append it back to the queue.
#         for i in range(len(digits)):
#             length = len(res_q)
#             for _ in range(length):
#                 s = res_q.popleft()
#                 for c in helper[dic[digits[i]] - 2]:
#                     res_q.append(s + c)
                    
#         return res_q
                
#   Runtime: 28 ms, faster than 86.01% of Python3 online submissions for Letter Combinations of a Phone Number.
#   Memory Usage: 13.9 MB, less than 99.99% of Python3 online submissions for Letter Combinations of a Phone Number.
#   By Fu


#18th problem
#status: solved but slow
# from itertools import permutations

# def allComb(string, lenght: int):
#     '''
#     nothing
#     '''
#     outlst = []
#     for comb in permutations(string, lenght):
#         outlst.append(comb)
#     outlst = list(map(list, outlst))

#     return sorted(outlst)


# class Solution:
# 	def fourSum(self, nums: list[int], target: int) -> list[list[int]]:
# 		triplets = allComb(nums, 4)
# 		validTriplets = [i for i in triplets if sum(i)==target]

# 		for j in range (0, len(validTriplets)):
# 			validTriplets[j] = sorted(validTriplets[j])

# 		validOut = []
# 		for i in validTriplets:
# 			if i not in validOut:
# 				validOut.append(i)

# 		return validOut

#alternativeve and faster

# class Solution:
#     def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
#         quadruplets = set()
#         hashtable = {}
        
#         for i in range(len(nums)):
#             for j in range(i + 1, len(nums)):
#                 currSum = nums[i] + nums[j]
#                 if target - currSum in hashtable:
#                     for pair in hashtable[target - currSum]:
#                         sortedPair = sorted([nums[i], nums[j], pair[0], pair[1]])
#                         quadruplets.add(tuple(sortedPair))
                        
#             for k in range(i):
#                 pairSum = nums[i] + nums[k]
#                 if pairSum not in hashtable:
#                     hashtable[pairSum] = [[nums[i], nums[k]]]
#                 else:
#                     hashtable[pairSum].append([nums[i], nums[k]])

#         return quadruplets


#19th problem
#status: solved
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# class Solution:
#     def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
#         first = second = head
#         for _ in range(n):
#             first = first.next
            
#         if not first:
#             return head.next
#         while first.next:
#             first = first.next
#             second = second.next
        
#         second.next = second.next.next
#         return head


#20th problem
#status: solved
# class Solution:
#     def isValid(self, s: str) -> bool:
#         brackets_open = ('(', '[', '{', '<')
#         brackets_closed = (')', ']', '}', '>')
#         stack = []
#         for i in s:
#             if i in brackets_open:
#                 stack.append(i)
#             if i in brackets_closed:    
#                 if len(stack) == 0:
#                     return False
#                 index = brackets_closed.index(i)
#                 open_bracket = brackets_open[index]
#                 if stack[-1] == open_bracket:
#                     stack = stack[:-1]  
#                 else: 
#                     return False  
#         return (not stack)


#21th problem
#status: solved
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# class Solution:
#     def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
# # when one of the lists is empty we simply return the other list
#         if not l1:
#             return l2
#         elif not l2:
#             return l1
        
#         # we need a placeholder node where we build our merged linked list
#         placeholder = ListNode(0)
#         # we need a pointer at its head to keep the track of it.
#         head = placeholder
        
#         while (l1 and l2):
#             if l1.val <= l2.val:
#                 placeholder.next = l1 # attach the smaller node to the dummy node
#                 l1 = l1.next # move the list forward
#             else:
#                 placeholder.next = l2
#                 l2 = l2.next
            
#             placeholder = placeholder.next # since a node is attached we need to increment the pointer
        
#         # at end there may be remaining items in any of the list. Attach them
#         if not l1:
#             placeholder.next = l2
#         else:
#             placeholder.next = l1
        
#         # head is pointing to the dummy node so we must return the actual node 
#         # dummy -> merged_linked_list
#         return head.next


#22th problem
#status: solved
# class Solution:
#     def generateParenthesis(self, n: int) -> list[str]:
#         result = []
#         def dfs(s, now):
#             cnt = s.count("(")
#             if len(s) == 2*n:
#                 return result.append(s)
#             if now == 0:
#                 dfs(s+"(", now+1 )
#             else :
#                 if cnt < n:
#                     dfs(s+"(",now+1)
#                 dfs(s+")",now-1)
#         dfs("(",1)
#         return result


#23th problem
#status: solved
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# class Solution:
#     def mergeKLists(self, lists):
#         newlists = []
#         for li in lists:
#             if li != None: 
#                 newlists.append(li)

#         if len(newlists) == 0:
#             return None
#         elif len(newlists) == 1:
#             return newlists[0]
#         else:
#             while(len(newlists) >= 2):
#                 list1 = newlists.pop()
#                 list2 = newlists.pop()
#                 pointer = ListNode(-1)
#                 pointer_head = pointer 
#                 while(list1 != None and list2 !=None):
#                     if list1.val <= list2.val:
#                         pointer.next = list1
#                         list1 = list1.next
#                     else:
#                         pointer.next = list2
#                         list2 = list2.next
#                     pointer = pointer.next

#                 if list1 == None:
#                     pointer.next = list2
#                 else:
#                     pointer.next = list1
#                 pointer = pointer.next
                
#                 newlists.insert(0, pointer_head.next)
                
#         return newlists[0]


#24th problem
#status: solved
# class Solution:
#     def swapPairs(self, head: ListNode) -> ListNode:
#             if not head or not head.next: #if list is empty or list.lenght() == 1
#                 return head
#             p1, p2 = head #pointers

#             while p1 != None:
#                 if not p1.next: #in case if list.lenght() is odd
#                     break
#                     p2 = p1.next #p2 is always must be next
#                     p1.val, p2.val, p1 = p2.val, p1.val, p1.next.next # for swapping   # also for updating p1
            
#             return head


#25th problem
#status: solved
# class Solution:
#     def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
#         if k == 1:
#             return head
#         pre_head = ListNode()
#         ph = pre_head
#         p = head
#         p2 = head
#         cnt = 0
#         while p:
#             p = p.next
#             cnt += 1
#             if cnt == k:
#                 cnt = 0
#                 end = p2
#                 prev = p2
#                 p2 = p2.next
#                 while p2 != p:
#                     tmp = p2.next
#                     p2.next = prev
#                     prev = p2
#                     p2 = tmp
#                 ph.next = prev
#                 ph = end
#         ph.next = p2
#         return pre_head.next


#26th problem
#status: solved
# class Solution:
#     def removeDuplicates(self, nums: list[int]) -> int:
#         i = 0
#         for _ in range (len(nums)-1):
#             if nums[i] == nums[i+1]:
#                 del nums[i]
#             else:
#                 i += 1
#         return len(nums)


#27th problem
#status: solved
# class Solution:
#     def removeElement(self, nums: List[int], val: int) -> int:
#         i = 0
#         for _ in range (len(nums)):
#             if nums[i] == val:
#                 del nums[i]
#             else:
#                 i += 1
#         return len(nums)


#28th problem
#status: solved
# class Solution:
#     def strStr(self, haystack: str, needle: str) -> int:
#         if haystack == '' and needle == '':
#             return 0
        
#         if needle not in haystack:
#             return -1

#         ln = len(needle)
#         for i in range(len(haystack)):
#             if haystack[i:i+ln] == needle:
#                 lst.append(i)
#         return -1


#29th problem
#status: solved
# import math
# class Solution:
#     def divide(self, dividend: int, divisor: int) -> int:
#         MAX = 2**31
#         sign = (dividend>=0 and divisor>0) or (dividend<=0 and divisor<0)
#         ans = math.floor(abs(dividend)/abs(divisor))
#         if (not sign and ans>MAX) or (sign and ans>(MAX -1)) :
#             ans = MAX - 1   
#         return ans if sign else -ans


#30th problem
#status: solved
# class Solution:
#     def findSubstring(self, s: str, words: List[str]) -> List[int]:
#         from collections import Counter

#         word_len = len(words[0])
#         words_len_sum, word_counter = word_len*len(words), Counter(words)

        
#         def substring_has_words(s):
#             return Counter([s[i:i+word_len] for i in range(0, len(s), word_len)]) == word_counter
        
#         res = []
#         for i in range(len(s)-words_len_sum+1):
#             if substring_has_words(s[i: i+words_len_sum]):
#                 res.append(i)
        
#         return res


#31th problem
#status: solved
# class Solution:
#     def nextPermutation(self, nums: list[int]) -> None:
#         """
#         Do not return anything, modify nums in-place instead.
#         """
        
#         length = len(nums)
        
#         # Edge case.
#         if length <=2:             
#             return nums.reverse()
          
#         j = length -2 
#         i = length -1
        
#         # Taking eg [1,5,2,4,3] 
        
#         # This loop is for finding index where ascending order from last index is break
#          # j will be at 2 index
#         while j>=0 and nums[j]>=nums[j+1]:
#             j-=1
        
#         # for cases such as [3,2,1]
#         if j==-1:
#             return nums.reverse()
        
#         # This loop is to find index of next big element than index j
#         # i will be at index 4 
#         while i>=0  and nums[i]<=nums[j]:
#             i-=1
        
                              
#         #after swapping [1,5,3,4,2]
#         nums[i],nums[j]= nums[j],nums[i]
        
#         #reversing element after swapping 
#         #[1,5,3,2,4]
#         nums[j+1:] = nums[:j:-1]  



#32th problem
#status: solved
# class Solution:
#     def longestValidParentheses(self, s: str) -> int:
#         l = r = 0
#         res = 0
#         for ch in s:
#             if ch == '(':
#                 l += 1
#             else:
#                 r += 1
#             if l == r:
#                 res = max(res, r * 2)
#             elif r > l:
#                 l = r = 0
#         l = r = 0
#         for ch in reversed(s):
#             if ch == ')':
#                 l += 1
#             else:
#                 r += 1
#             if l == r:
#                 res = max(res, r * 2)
#             elif r > l:
#                 l = r = 0
#         return res


# 33th problem 
# status: solved
# class Solution:
#     def search(self, nums: list[int], target: int) -> int:
#         if target in nums:
#             return nums.index(target)
#         return -1


# 34th problem
# status: solved
# class Solution:
#     def searchRange(self, nums: List[int], target: int) -> List[int]:
#         # find index of the target
#         left,right = 0,len(nums)-1
#         while left <= right:
#             mid = (left+right)//2
#             if nums[mid] == target:
#                 # to find left index
#                 leftindex = mid
#                 while leftindex >= 0 and nums[leftindex] == target:
#                     leftindex -= 1
#                 # to find right index
#                 rightindex = mid
#                 while rightindex < len(nums) and nums[rightindex] == target:
#                     rightindex += 1
#                 return [leftindex+1,rightindex-1]
#             elif nums[mid] < target:
#                 left = mid + 1
#             else:
#                 right = mid -1
#         return [-1,-1]


# 35th problem 
# status: solved
# class Solution:
#     def searchInsert(self, nums: list[int], target: int) -> int:
#         nums.append(target)
#         nums.sort()
#         return nums.index(target)


# 36th problem
# status: solved
# def isValid(lst):
#     for ls in lst:
#         l = set(ls)
#         l.discard('.')
#         while l:
#             if ls.count(l.pop())>1:
#                 return False
#     return True
# class Solution:
#     def isValidSudoku(self, board: list[list[str]]) -> bool:
#         squares = []
#         for k in range(0,9,3):
#             for i in range(0,9,3):
#                 temp = []
#                 temp0 = board[i:i+3]
#                 for j in temp0:
#                     temp += j[k:k+3]
#                 squares.append(temp)

#         columns = []
#         for q in range(9):
#             columns.append([b[q] for b in board])


#         if not isValid(squares):
#             return False

#         if not isValid(board):
#             return False

#         if not isValid(columns):
#             return False


#         return True


# 37th problem 
# status: solved
# class Solution:
#     def solveSudoku(self, A: list[list[str]]) -> None:
#         """
#         Do not return anything, modify board in-place instead.
#         """
        
#         self.fin=[]         #To store final result
        
#         #Sets to keep track of invalid number
#         right = [set() for _ in range(9)]                               #Horizontally
#         down = [set() for _ in range(9)]                                #Vertically
#         cube = [[set() for q in range(3)] for _ in range(3)]            #Sub-block
        
        
#         lis = [[0]*9 for _ in range(9)]
        
#         pos = []          #to store places we need to traverse
        
#         #To store int values in list named "lis" and 0 in place of "."
#         for i in range(9):
#             for j in range(9):
#                 if A[i][j]=='.':
#                     pos.append((i,j))
#                 else:
#                     right[i].add(int(A[i][j]))
#                     down[j].add(int(A[i][j]))
#                     cube[i//3][j//3].add(int(A[i][j]))
#                     lis[i][j] = int(A[i][j])
        
        
#         #BackTrack
#         def trav(curr,left):
#             if left==[]:
#                 # When everything is placed perfectly
#                 self.fin=curr
#                 return 1
                
#             i,j = left[0]         #block that we'll be working on
            
#             for val in range(1,10):
#                 if val not in right[i] and val not in down[j] and val not in cube[i//3][j//3]:
                
#                     #Add the temporary 'val' in all required sets and lists
#                     curr[i][j]=val
#                     right[i].add(val)
#                     down[j].add(val)
#                     cube[i//3][j//3].add(val)
                    
#                     #Traverse further
#                     ret = trav(curr,left[1:])
#                     if ret==1:
#                         #when we have already found the required sudoku
#                         return 1
#                     #Removing the temporary 'val' as it wasn't the desired one
#                     curr[i][j]=0
#                     right[i].remove(val)
#                     down[j].remove(val)
#                     cube[i//3][j//3].remove(val)
            
#             return -1
        
#         trav(lis,pos)
        
#         #Since we need to modify in place, thus putting the new values to the main list
#         for i in range(9):
#             A[i] = list(map(str,self.fin[i]))
            
#         return A


# 38th problem
# status: solved
# class Solution:
#     def countAndSay(self, n: int) -> str:
#         res = "1"
#         for i in range(1, n):
#             count = 1
#             new_res = ""
#             for j in range(0, len(res)-1):
#                 if res[j] != res[j+1]:
#                     new_res += f"{count}{res[j]}"
#                     count = 1
#                 else:
#                     count += 1
#             new_res += f"{count}{res[len(res)-1]}"
#             res = new_res
#         return res


# 39th problem
# status: solved
# from collections import defaultdict
# class Solution:
#     def combinationSum(self, candidates: list[int], target: int) -> list[list[int]]:
#         # store all combinations according to their sum
#         cache = defaultdict(list)
        
#         for c in candidates:
#             if c > target:
#                 continue
            
#             # store candidate
#             cache[c].append([c])
            
#             # get the combinations of all values that candidate can add to and be < target
#             for i in range(1, target-c+1):
#                 seen_list = cache[i]
#                 for seen in seen_list:
#                     # store new combinations
#                     cache[c+i].append(seen+[c])  
    
#         # return combinations that add to target value            
#         return cache[target]


# 40th problem
# stauts: solved
# class Solution:
#     def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
#         def dfs( candidates: list[int], target: int, cur_path: list[int], ans: list[list[list[int]]] ):
#             if target < 0: return
#             if target == 0:
#                 ans.append( cur_path )
#                 return
#             for index, num in enumerate(candidates):
#                 if index > 0 and candidates[index] == candidates[index - 1]: continue
#                 if num > target: continue
#                 dfs( candidates[index + 1:], target - num, cur_path + [num], ans )
                
#         ans = []
#         dfs ( sorted([x for x in candidates if x <= target]), target, [], ans )
#         return ans