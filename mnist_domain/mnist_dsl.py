def get_grammar(symb):
    rules = []

    # higher-order combinators
    rules.append(("list-int", "map", "int->int", "list-int"))
    rules.append(("list-int", "filter", "int->bool", "list-int"))
    rules.append(("int", "fold", "int->int->int", "const-int", "list-int"))
    rules.append(("int", "length", "list-int"))
    # rules.append(("list-int", "slice", "list-int", "const-int", "const-int"))

    # higher-order combinators for images
    # rules.append(("list-pimg-int", "map", "img-int->pimg-int", "list-img-int"))
    rules.append(("list-int", "map_imgs", "pimg-int->int", "list-pimg-int"))

    # function application for images
    # rules.append(("pimg-int", "apply", "img-int->pimg-int", "img-int"))
    rules.append(("const-int", "apply", "pimg-int->int", "pimg-int"))

    # function application (currying)
    rules.append(("int->int", "curry", "int->int->int", "const-int"))
    rules.append(("int->bool", "curry", "int->int->bool", "const-int"))

    # functions
    rules.append(("int->int->int", "plus"))
    rules.append(("int->int->int", "max"))
    rules.append(("int->int->int", "mult"))

    rules.append(("int->int->bool", "ge"))
    rules.append(("int->int->bool", "le"))

    # functions for images
    # rules.append(("img-int->pimg-int", "id"))
    rules.append(("pimg-int->int", "pred_int"))

    # inputs
    rules.append(("list-pimg-int", "input-list"))
    rules.append(("pimg-int", "input-img"))

    # constants
    rules.append(("const-int", "0"))
    rules.append(("const-int", "1"))
    rules.append(("const-int", "2"))
    rules.append(("const-int", "3"))
    rules.append(("const-int", "4"))
    rules.append(("const-int", "5"))
    rules.append(("const-int", "6"))
    rules.append(("const-int", "7"))
    rules.append(("const-int", "8"))
    rules.append(("const-int", "9"))

    return Grammar(rules, symb)


class Grammar:
    def __init__(self, rules, start):
        self.rules = {}
        for rule in rules:
            if rule[0] not in self.rules:
                self.rules[rule[0]] = []
            self.rules[rule[0]].append(rule[1:])
        self.start = start


def parse_helper(toks):
    name = toks.pop()
    if "(" in name or ")" in name:
        raise Exception()
    children = []
    while True:
        tok = toks.pop()
        if tok == ")":
            return Expr(name, children)
        elif tok == "(":
            children.append(parse_helper(toks))
        else:
            children.append(Expr(tok))


class Expr:
    def __init__(self, name, children=[]):
        self.name = name
        self.children = children

    def str_id(self, ident="0"):
        return f"({ident} {self.name}{"".join([" " + child.str_id(ident + "-{}".format(i)) for i, child in enumerate(self.children)])})"

    def __str__(self):
        if len(self.children) == 0:
            return self.name
        else:
            return f"({self.name}{"".join([" " + str(child) for child in self.children])})"
        
    def duplicate(self):
        return self
    
    def get_grammar_rule(self):
        return self.name