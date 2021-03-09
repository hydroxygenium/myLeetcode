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
#status: not solved
class Solution:
    def longestCommonPrefix(self, strs: list[str]) -> str:
        pass