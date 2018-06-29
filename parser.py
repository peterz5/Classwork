import nltk

from collections import defaultdict
from nltk.ccg.chart import printCCGDerivation
from nltk.ccg.lexicon import Token

from utils import *

class CCGParser:

	""" Parse CCG according to the CKY algorithm. """

	DEFAULT_START = AtomicCategory("S")
	DEFAULT_RULE_SET = []

	def __init__(self, lexicon, rules=DEFAULT_RULE_SET):
		self.lexicon = lexicon
		self.rules = rules

	@staticmethod
	@rule(DEFAULT_RULE_SET, 2, "A")
	def application(cat1, cat2):
		"""
		Implements the combination rule for function application.
		If cat1 and cat2 can be left or right-applied to each other (assuming cat1 is left of cat2),
		return the resulting category. Otherwise, return None.
		Hints:
			* isinstance(cat, CombinedCategory) tells you whether a category is combined or atomic.
			* For a combined category, cat.left and cat.right give the left and right subcategories.
			* For a combined category, cat.direction is either "\\" or "/".
			* You can use == to check whether categories are the same
		"""

		if isinstance(cat1, CombinedCategory) and isinstance(cat2, AtomicCategory):
			if cat1.direction == '/':
				if cat2 == cat1.right:
					return cat1.left

		elif isinstance(cat1, AtomicCategory) and isinstance(cat2, CombinedCategory):
			if cat2.direction == '\\':
				if cat1 == cat2.right:
					return cat2.left
		
		elif isinstance(cat1, CombinedCategory) and isinstance(cat2, CombinedCategory):
			if cat1 == cat2.right:
				if cat2.direction == '\\':
					return cat2.left

			elif cat1.right == cat2:
				if cat1.direction == '/':
					return cat1.left

		return None


	@staticmethod
	@rule(DEFAULT_RULE_SET, 2, "C")
	def composition(cat1, cat2):
		"""
		Implements the combination rule for function composition.
		If cat1 and cat2 can be left or right-composed, return the resulting category.
		Otherwise, return None.
		"""
		if isinstance(cat1, CombinedCategory) and isinstance(cat2, CombinedCategory):	
			
			if cat1.right == cat2.left and cat1.direction == '/':
				cat = CombinedCategory(cat1.left, cat2.direction, cat2.right)		
				return cat
			elif cat1.left == cat2.right and cat2.direction == '\\':
				cat = CombinedCategory(cat2.left, cat1.direction, cat1.right)
				return cat

		return None

	@staticmethod
	@rule(DEFAULT_RULE_SET, 1, "T")
	def typeRaising(cat1, cat2):
		"""
		Implements the combination rule for type raising.
		If cat2 satisfies the type-raising constraints, type-raise cat1 (and vice-versa).
		Return value when successful should be the tuple (cat, dir):
			* cat is the resulting category of the type-raising
			* dir is "/" or "\\" and represents the direction of the raising
			* If no type-raising is possible, return None
		Hint: use cat.innermostFunction() to implement the conditional checks described in the
			specification.
		"""

		#cat = CombinedCategory(cat1, "/", cat2)

		if isinstance(cat1, AtomicCategory) and isinstance(cat2, CombinedCategory):
			if isinstance(cat2.innermostFunction().right, AtomicCategory):
				cat = CombinedCategory(cat2.innermostFunction().left, '/', cat2.innermostFunction())
				
				if cat2.innermostFunction().direction == '/' or not cat2.innermostFunction().right == cat1:
					return None
				
				return cat, cat.direction

		if isinstance(cat2, AtomicCategory) and isinstance(cat1, CombinedCategory):
			if isinstance(cat1.innermostFunction().right, AtomicCategory):
				cat = CombinedCategory(cat1.innermostFunction().left, '\\', cat1.innermostFunction())
				
				if cat1.innermostFunction().direction == '\\' or not cat1.right == cat2:
					return None
				
				return cat, cat.direction

		return None


	class VocabException(Exception):
		pass

	def fillParseChart(self, tokens):
		"""
		Builds and fills in a CKY parse chart for the sentence represented by tokens.
		The argument tokens is a list of words in the sentence.
		Each entry in the chart should be a list of Constituents:
			* Use AtomicConstituent(cat, word) to construct initialize Constituents of words.
			* Use CombinedConstituent(cat, leftPtr, rightPtr, rule) to construct Constituents
			  produced by rules. leftPtr and rightPtr are the Constituent objects that combined to
			  form the new Constituent, and rule should be the rule object itself.
		Should return (chart, parses), where parses is the final (top right) entry in the chart. 
		Each tuple in parses corresponds to a parse for the sentence.
		Hint: initialize the diagonal of the chart by looking up each token in the lexicon and then
			use self.rules to fill in the rest of the chart. Rules in self.rules are sorted by
			increasing arity (unary or binary), and you can use rule.arity to check the arity of a
			rule.
		"""

		chart = defaultdict(list)



		for i in range (1, len(tokens)+1):
			for j in self.lexicon.getCategories(tokens[i-1]):
				chart[(i-1, i)].append(AtomicConstituent(j, tokens[i-1]))
		
		for j in range (2, len(tokens)+1):
			for i in range(j-2, -1, -1):
				for k in range(i+1, j):		
					if chart[(i,k)] != None and chart[(k, j)] != None:
						for x in chart[(i, k)]:
							for y in chart[(k,j)]:
								for rule in self.rules:
									
									if isinstance(x.cat, AtomicCategory) and isinstance(y.cat, CombinedCategory):
				 						
										result = rule(x.cat, y.cat)
										
										if result and rule.arity == 2:
										chart[(i,j)].append(CombinedConstituent(result, [x, y], rule))
										elif result and rule.arity == 1:	
										chart[(i,k)].append(CombinedConstituent(result[0], [x], rule))

									elif isinstance(x.cat, CombinedCategory) and isinstance(y.cat, AtomicCategory):
										
										result = rule(x.cat, y.cat)
										
										if result and rule.arity == 2:
										chart[(i,j)].append(CombinedConstituent(result, [x, y], rule))
										elif result and rule.arity == 1:
										chart[(k,j)].append(CombinedConstituent(result[0], [y], rule))

									elif rule.arity == 2:
										if isinstance(x.cat, CombinedCategory) and isinstance(y.cat, CombinedCategory):
										result = rule(x.cat, y.cat)
										if isinstance(result, Category):
										chart[(i,j)].append(CombinedConstituent(result, [x, y], rule))

		parses = chart[(0,len(tokens))]

		return (chart, parses)


	@staticmethod
	def generateParseTree(cons, chart):
		"""
		Helper function that returns an NLTK Tree object representing a parse.
		"""
		token = Token(None, cons.cat, None)
		if isinstance(cons, AtomicConstituent):
			return nltk.tree.Tree(
				(token, u"Leaf"),
				[nltk.tree.Tree(token, [cons.word])]
			)
		else:
			if cons.rule == CCGParser.typeRaising:
				return nltk.tree.Tree(
					(token, cons.rule.name),
					[CCGParser.generateParseTree(cons.ptrs[0], chart)]
				)
			else:
				return nltk.tree.Tree(
					(token, cons.rule.name),
					[CCGParser.generateParseTree(cons.ptrs[0], chart),
					CCGParser.generateParseTree(cons.ptrs[1], chart)]
				)

	def getParseTrees(self, tokens):
		"""
		Reconstructs parse trees for the sentences by following backpointers.
		"""
		chart, parses = self.fillParseChart(tokens)
		for cons in parses:
			yield CCGParser.generateParseTree(cons, chart)

	def accepts(self, tokens, sentCat=DEFAULT_START):
		"""
		Return True iff the sentence represented by tokens is in this language (i.e. has at least
			one valid parse).
		"""
		_, parses = self.fillParseChart(tokens)
		
		for cons in parses:
			if cons.cat == sentCat: return True
		return False