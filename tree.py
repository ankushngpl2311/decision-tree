from collections import deque 


class node:
	def __init__(self):
		self.children ={}   # {low:pointer to low,high:pointer to high}
	def insert(self,name,positive,negative):
		self.name= name
		self.positive=positive
		self.negative=negative
		

	# def insert(self,obj):
	# 	self.children.append(obj)



# def preorderTraversal(root): 
  
#     for i in root.children:
#     	preorderTraversal(root.children[i])
#     	print("node name= ",root.name)
#     	print("positive= ",root.positive)
#     	print("negative= ",root.negative)
#     	preorderTraversal(root.ch)

def postorder(root):
    """
    :type root: Node
    :rtype: List[int]
    """
    if root is None:
        return []
    
    stack, output = [root, ], []
    while stack:
        root = stack.pop()
        if root is not None:
            output.append(root.name)
        for c in root.children:
            stack.append(root.children[c])
            
    return output[::-1]
	