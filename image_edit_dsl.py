from typing import List


class Node:
    def __str__(self):
        return type(self).__name__

    def __eq__(self, other):
        return isinstance(other, Node)

    def __lt__(self, other):
        self_str = str(self)
        other_str = str(other)
        if self_str == "Hole" and other_str == "Hole":
            return self_str < other_str
        elif self_str == "Hole":
            return False
        elif other_str == "Hole":
            return True
        elif "Hole" in self_str and "Hole" in other_str:
            return self_str < other_str
        elif "Hole" in other_str:
            return True
        elif "Hole" in self_str:
            return False
        return self_str < other_str


class Expression(Node):
    def __init__(self, val=None, output_under=None, output_over=None):
        self.abs_value = None

    def __eq__(self, other):
        return str(self) == str(other)


class Map(Expression):
    def __init__(
        self,
        expression: Expression,
        restriction,
        position,
        val=None,
        output_under=None,
        output_over=None,
    ):
        super().__init__(val, output_under, output_over)
        self.expression = expression
        self.restriction = restriction
        self.position = position

    def __str__(self):
        return (
            type(self).__name__
            + "("
            + str(self.expression)
            + ", "
            + str(self.restriction)
            + ", "
            + str(self.position)
            + ")"
        )
    
    def get_grammar_rule(self):
        return "Map_{}".format(self.position)

    def duplicate(self):
        return Map(
            self.expression.duplicate(),
            self.restriction.duplicate(),
            self.position,
        )

    def __eq__(self, other):
        if not isinstance(other, Map):
            return False
        return (
            self.expression == other.expression
            and self.restriction == other.restriction
            and self.position == other.position
        )


class Attribute(Expression):
    pass


class IsObject(Attribute):
    def __init__(self, obj: str, val=None, output_under=None, output_over=None):
        super().__init__(val, output_under, output_over)
        self.obj = obj

    def __str__(self):
        return type(self).__name__ + "(" + str(self.obj) + ")"

    def __eq__(self, other):
        if isinstance(other, IsObject):
            return self.obj == other.obj
        return False

    def duplicate(self):
        return IsObject(self.obj)
    
    def get_grammar_rule(self):
        return type(self).__name__ + "_" + str(self.obj)



class IsSmiling(Attribute):
    def duplicate(self):
        return IsSmiling()

    def __str__(self):
        return "Smile"
    
    def get_grammar_rule(self):
        return type(self).__name__


class EyesOpen(Attribute):
    def duplicate(self):
        return EyesOpen()

    def __str__(self):
        return "EyesOpen"
    
    def get_grammar_rule(self):
        return type(self).__name__


class MouthOpen(Attribute):
    def duplicate(self):
        return MouthOpen()

    def __str__(self):
        return "MouthOpen"
    
    def get_grammar_rule(self):
        return type(self).__name__




class Union(Expression):
    def __init__(
        self, expressions: List[Expression], val=None, output_under=None, output_over=None
    ):
        super().__init__(val, output_under, output_over)
        self.expressions = expressions

    def __str__(self):
        expression_strs = [str(extr) for extr in self.expressions]
        return type(self).__name__ + "[" + ", ".join(expression_strs) + "]"

    def duplicate(self):
        return Union([extr.duplicate() for extr in self.expressions])

    def __eq__(self, other):
        if isinstance(other, Union):
            return self.expressions == other.expressions
        return False
    
    def get_grammar_rule(self):
        return "Union_{}".format(len(self.expressions)) 


class Intersection(Expression):
    def __init__(
        self, expressions: List[Expression], val=None, output_under=None, output_over=None
    ):
        super().__init__(val, output_under, output_over)
        self.expressions = expressions

    def __str__(self):
        expression_strs = [str(extr) for extr in self.expressions]
        return type(self).__name__ + "[" + ", ".join(expression_strs) + "]"

    def duplicate(self):
        return Intersection(
            [extr.duplicate() for extr in self.expressions]
        )

    def __eq__(self, other):
        if isinstance(other, Intersection):
            return self.expressions == other.expressions
        return False
    
    def get_grammar_rule(self):
        return "Intersection_{}".format(len(self.expressions)) 


class Complement(Expression):
    def __init__(
        self, expression: Expression, val=None, output_under=None, output_over=None
    ):
        super().__init__(val, output_under, output_over)
        self.expression = expression

    def __str__(self):
        return type(self).__name__ + "(" + str(self.expression) + ")"

    def duplicate(self):
        return Complement(self.expression.duplicate())

    def __eq__(self, other):
        if isinstance(other, Complement):
            return self.expression == other.expression
        return False
    
    def get_grammar_rule(self):
        return "Complement"


class Position(Node):
    def __eq__(self, other):
        return type(self) is type(other)



class GetLeft(Position):
    pass


class GetRight(Position):
    pass


class GetBelow(Position):
    pass


class GetAbove(Position):
    pass


class GetContains(Position):
    pass


class GetIsContained(Position):
    pass

