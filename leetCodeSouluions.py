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


#15th problem
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


#16th problem
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


#17th problem
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