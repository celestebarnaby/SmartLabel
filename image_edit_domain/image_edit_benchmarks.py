from image_edit_domain.image_edit_dsl import *
from benchmark import Benchmark

# THESE ARE THE IMAGEEYE BENCHMARKS!!
image_edit_benchmarks = benchmarks = [
    Benchmark(
        Map(
            Union([MouthOpen(), IsSmiling(), EyesOpen()]),
            IsObject("Person"),
            GetBelow(),
        ),
        "Bodies of faces that have glasses, are smiling, or have eyes open",
        "wedding",
    ),
    Benchmark(
        Union(
            [Map(IsObject("Bride"), IsObject("Face"), GetRight()), Map(IsObject("Bride"), IsObject("Face"), GetLeft())]
        ),
        "Faces to the left and right of bride.",
        "wedding",
    ),
    Benchmark(
        Union([IsObject("Bride"), Map(IsObject("Bride"), IsObject("Face"), GetAbove())]),
        "Face 8 and face 34 when it is behind face 8",
        "wedding",
    ),
    Benchmark(
        Map(IsObject("Face"), IsObject("Face"), GetAbove()),
        "All faces in back",
        "wedding",
    ),
    Benchmark(
        Intersection([Complement(IsSmiling()), Map(IsObject("Face"), IsObject("Face"), GetAbove())]),
        "All faces in back that are not smiling",
        "wedding",
    ),
    Benchmark(
        Intersection(
            [Map(IsObject("Face"), IsObject("Face"), GetRight()), Map(IsObject("Face"), IsObject("Face"), GetLeft())]
        ),
        "All faces except leftmost and rightmost face",
        "wedding",
    ),
    Benchmark(
        Intersection([IsSmiling(), EyesOpen()]),
        "All faces that are smiling and have eyes open",
        "wedding",
    ),
    Benchmark(
        Intersection([IsObject("Face"), Complement(Intersection([IsSmiling(), EyesOpen()]))]),
        "All faces that are not smiling and have eyes open",
        "wedding",
    ),
    Benchmark(
        Intersection([IsSmiling(), EyesOpen(), Complement(MouthOpen())]),
        "All faces that are smiling and have eyes open, except faces with mouth open.",
        "wedding",
    ),
    Benchmark(
        Union([Map(IsObject("Bride"), IsObject("Face"), GetAbove()), Map(IsObject("Suit"), IsObject("Face"), GetAbove())]),
        "Face of the bride and faces of people wearing suits",
        "wedding",
    ),
    Benchmark(
        Intersection([IsObject("Face"), Complement(IsSmiling())]),
        "All faces except faces that are smiling",
        "wedding",
    ),
    Benchmark(
        Union([IsObject("Glasses"), Intersection([IsSmiling(), EyesOpen()])]),
        "Glasses, plus faces that are smiling and have eyes open",
        "wedding",
    ),
    Benchmark(
        Map(Map(IsObject("Face"), IsObject("Face"), GetRight()), IsObject("Face"), GetRight()),
        "All faces except 2 leftmost faces",
        "wedding",
    ),
    Benchmark(
        Union([EyesOpen(), Intersection([IsObject("Face"), Complement(MouthOpen())])]),
        "Faces that are not smiling or have eyes open",
        "wedding",
    ),
    Benchmark(
        Union([IsSmiling(), Map(IsSmiling(), IsObject("Face"), GetRight())]),
        "Smiling faces and faces directly to their right",
        "wedding"
    ),

    Benchmark(
        Union(
            [
                IsObject("Bride"),
                Map(IsObject("Bride"), IsObject("Face"), GetLeft()),
                Map(IsObject("Bride"), IsObject("Face"), GetRight()),
            ]
        ),
        "Bride and faces directly to their left and right",
        "wedding",
    ),
    
    Benchmark(
        Union([IsObject("Car"), IsObject("Bicycle")]),
        "All cars and bicycles",
        "cars",
    ),
    Benchmark(
        Map(Intersection([IsSmiling(), EyesOpen()]), IsObject("Car"), GetIsContained()),
        "Cars with visible wheels",
        "cars",
    ),
    Benchmark(
        Map(IsObject("Car"), IsObject("Face"), GetContains()),
        "All faces within car",
        "cars",
    ),
    Benchmark(
        Complement(IsObject("Car")),
        "All objects except cars",
        "cars",
    ),
    Benchmark(
        Complement(Union([IsObject("Car"), IsObject("Bicycle")])),
        "All objects except cars and bicycles",
        "cars",
    ),
    Benchmark(
        Map(IsObject("Car"), Union([IsObject("Person"), IsObject("Bicycle")]), GetContains()),
        "All wheels on a car",
        "cars",
    ),
    Benchmark(
        Intersection(
            [EyesOpen(), Complement(Map(IsObject("Car"), EyesOpen(), GetContains()))]
        ),
        "All people with eyes open not in a car",
        "cars",
    ),
    Benchmark(
        Map(IsObject("Person"), IsObject("Car"), GetIsContained()),
        "Cars with person in front of them",
        "cars",
    ),
    Benchmark(
        Map(Map(IsObject("Helmet"), IsObject("Person"), GetBelow()), IsObject("Bicycle"), GetBelow()),
        "Bicycles that are being ridden",
        "cars",
    ),
    Benchmark(
        Intersection(
            [
                IsObject("Bicycle"),
                Complement(Map(IsObject("Person"), IsObject("Bicycle"), GetBelow())),
            ]
        ),
        "Bicycles that are not being ridden",
        "cars",
    ),
    Benchmark(
        Intersection(
            [
                IsObject("Bicycle"),
                Complement(Map(EyesOpen(), IsObject("Bicycle"), GetBelow())),
            ]
        ),
        "Bicycles that are not being ridden by a person with eyes open",
        "cars",
    ),
    Benchmark(
        Union([IsObject("Bicycle"), IsObject("Car"), IsObject("Person")]),
        "All bicycles, cars, and people",
        "cars",
    ),
    Benchmark(
        Map(IsObject("Bicycle"), Union([IsSmiling(), EyesOpen()]), GetAbove()),
        "All people with mouth open on a bicycle",
        "cars",
    ),
    Benchmark(
        Intersection(
            [IsObject("Face"), Complement(Map(IsObject("Bicycle"), IsObject("Face"), GetAbove()))]
        ),
        "Faces of people not riding bicycles",
        "cars",
    ),
    Benchmark(
        Union([IsObject("Cat"), IsObject("Face")]),
        "All cats and human faces",
        "cats",
    ),
    Benchmark(
        Union([IsObject("Cat"), EyesOpen()]),
        "All cats and faces with eyes open",
        "cats",
    ),
    Benchmark(
        Intersection(
            [
                IsObject("Cat"),
                Complement(Map(IsObject("Cat"), IsObject("Cat"), GetBelow())),
            ]
        ),
        "Topmost cat",
        "cats",
    ),
    Benchmark(
        Intersection(
            [
                Map(IsObject("Cat"), IsObject("Cat"), GetRight()),
                Map(IsObject("Cat"), IsObject("Cat"), GetLeft()),
            ]
        ),
        "Cats that are between two other cats",
        "cats",
    ),
    Benchmark(
        Map(IsObject("Guitar"), IsObject("Face"), GetAbove()),
        "People playing guitar",
        "guitars",
    ),
    Benchmark(
        Union([IsObject("Guitar"), Map(IsObject("Guitar"), IsObject("Face"), GetAbove())]),
        "Guitars and people playing guitar",
        "guitars",
    ),
    Benchmark(
        Intersection(
            [IsObject("Face"), Complement(Map(IsObject("Guitar"), IsObject("Face"), GetAbove()))]
        ),
        "Faces of people not playing guitar",
        "guitars",
    ),

    # # RECEIPTS BENCHMARKS!
    # Benchmark(
    #     Map(MatchesWord("TOTAL"), IsObject("text"), GetRight()),
    #     'Text to the right of the word "TOTAL"',
    #     "receipts",
    # ),
    # Benchmark(
    #     Map(Map(MatchesWord("TOTAL"), IsPrice(), GetRight()), IsPrice(), GetAbove()),
    #     "Price that is above the total price",
    #     "receipts",
    # ),
    # Benchmark(
    #     Complement(Map(Map(IsObject("text"), IsObject("text"), GetAbove()), IsObject("text"), GetAbove())),
    #     "Bottom two columns of text",
    #     "receipts",
    # ),
    # Benchmark(
    #     Intersection([IsObject("text"), Complement(Union([MatchesWord("total"), IsPrice()]))]),
    #     "All text except prices and the word 'total'",
    #     "receipts",
    # ),
    # Benchmark(
    #     Map(MatchesWord("TOTAL"), IsPrice(), GetRight()),
    #     'Prices to the right of the word "TOTAL"',
    #     "receipts",
    # ),
    # Benchmark(
    #     Map(MatchesWord("tax"), IsObject("text"), GetAbove()),
    #     'Text above the word "tax"',
    #     "receipts",
    # ),
    # Benchmark(
    #     Union([IsPrice(), IsPhoneNumber()]),
    #     "All prices and all phone numbers",
    #     "receipts",
    # ),
    # Benchmark(
    #     Intersection([IsObject("text"), Complement(IsPrice())]),
    #     "All text that is not a price",
    #     "receipts",
    # ),
    # Benchmark(
    #     Union(
    #         [
    #             Map(MatchesWord("TOTAL"), IsObject("text"), GetRight()),
    #             Map(MatchesWord("SUBTOTAL"), IsObject("text"), GetRight()),
    #         ]
    #     ),
    #     'Text to the right of the word "TOTAL" or the word "SUBTOTAL"',
    #     "receipts",
    # ),
    # Benchmark(
    #     Map(IsPrice(), IsObject("text"), GetLeft()),
    #     "Text to the left of a price",
    #     "receipts",
    # ),
    # Benchmark(
    #     Intersection([IsObject("text"), Complement(Union([IsPrice(), IsPhoneNumber()]))]),
    #     "All text that is not a price or phone number",
    #     "receipts",
    # ),
    # Benchmark(
    #     Intersection(
    #         [IsPrice(), Complement(Map(MatchesWord("TOTAL"), IsObject("text"), GetRight()))]
    #     ),
    #     'Text that is a price and is not to the right of the word "TOTAL"',
    #     "receipts",
    # ),
    # Benchmark(
    #     Map(Map(IsObject("text"), IsObject("text"), GetLeft()), IsObject("text"), GetLeft()),
    #     "All text except two leftmost columns",
    #     "receipts"
    # ),
]


# THESE ARE THE POPL PAPER BENCHMARKS
# image_edit_benchmarks = [
#     Benchmark(
#         Intersection([IsSmiling(), Complement(MouthOpen())]),
#         "All faces that are smiling and do not have mouth open",
#         "wedding",
#     ),
#     Benchmark(
#         Map(IsObject("Guitar"), IsObject("Face"), GetAbove()),
#         "Faces playing guitar",
#         "festival"
#     ),
#     Benchmark(
#         Intersection([Complement(EyesOpen()), Map(IsObject("Guitar"), IsObject("Face"), GetAbove())]),
#         "Faces with eyes closed playing guitar",
#         "festival"
#     ),
#     Benchmark(
#         Map(IsObject("Car"), Intersection([IsObject("Face"), Complement(IsSmiling())]), GetContains()),
#         "People who aren't smiling inside cars",
#         "cars"
#     ),
#     Benchmark(
#        Union([IsObject("Bicycle"), Map(IsObject("Face"), IsObject("Car"), GetIsContained())]),
#         "Bicycles and cars with people in them",
#         "cars"
#     ),
#     Benchmark(
#         Union([IsObject("Person"), IsObject("Bicycle"), IsObject("Car")]),
#         "All bicycles, cars, and people",
#         "cars",
#     ),
#     Benchmark(
#         Intersection([IsSmiling(), EyesOpen(), Complement(MouthOpen())]),
#         "Faces that are smiling, have eyes open, and mouth closed",
#         "wedding",
#     ),

#     Benchmark(
#         Intersection([IsObject("Face"), Complement(IsSmiling()), Complement(IsObject("Bride"))]),
#         "All faces that are not smiling and are not the bride's face",
#         "wedding",
#     ),
#      Benchmark(
#         Union([Map(IsObject("Suit"), IsObject("Face"), GetAbove()), IsObject("Bride"), IsObject("Wedding Gown")]),
#         "Faces of people wearing suits, the bride, and the wedding gown",
#         "wedding",
#     ),
#     Benchmark(
#         Map(Map(IsObject("Face"), IsObject("Face"), GetRight()), IsObject("Face"), GetRight()),
#         "All faces except 2 leftmost faces",
#         "wedding",
#     ),
#     Benchmark(
#         Union([IsObject("Guitar"), Map(IsObject("Guitar"), IsObject("Face"), GetAbove())]),
#         "Guitars and people playing guitar",
#         "festival",
#     ),
#     Benchmark(
#         Map(Union([IsObject("Guitar"), IsSmiling()]), IsSmiling(), GetLeft()),
#         "Smiling faces to the left of guitars or other smiling faces",
#         "festival"
#     ),
#     Benchmark(
#         Map(Union([IsObject("Guitar"), IsObject("Microphone"), IsObject("Speaker")]), IsSmiling(), GetAbove()),
#         "Smiling faces to the left of guitars or microphones",
#         "festival"
#     ),
#     Benchmark(
#         Union([IsObject("Bicycle"), Map(IsObject("Helmet"), IsObject("Person"), GetBelow())]),
#         "Bicycles and people wearing helmets",
#         "cars",
#     ),
#     Benchmark(
#         Map(Map(IsObject("Helmet"), IsObject("Person"), GetBelow()), IsObject("Bicycle"), GetBelow()),
#         "Bicycles ridden by people wearing helmets",
#         "cars",
#     ),
#     Benchmark(
#         Map(Map(IsObject("Car"), IsObject("Bicycle"), GetRight()), IsObject("Person"), GetAbove()),
#         "People on bicycles next to cars",
#         "cars",
#     ),
#     Benchmark(
#         Map(Union([IsObject("Bicycle"), IsObject("Car"), IsObject("Bus")]), IsObject("Person"), GetRight()),
#         "People to the right of bicycles, cars, or buses",
#         "cars",
#     ),
#     Benchmark(
#         Intersection(
#             [IsObject("Person"), Complement(Map(IsObject("Bicycle"), IsObject("Person"), GetAbove()))]
#         ),
#         "People not riding bicycles",
#         "cars",
#     ),
#     Benchmark(
#         Intersection(
#             [IsObject("Face"), Complement(Map(IsObject("Car"), IsObject("Face"), GetContains()))]
#         ),
#         "People not in cars",
#         "cars",
#     ),
#     Benchmark(
#         Intersection(
#             [IsObject("Car"), Complement(Map(IsObject("Face"), IsObject("Car"), GetIsContained()))]
#         ),
#         "Cars without people in them",
#         "cars",
#     ),
#     Benchmark(
#         Intersection(
#             [
#                 IsObject("Face"),
#                 Complement(Map(IsObject("Face"), IsObject("Face"), GetBelow())),
#             ]
#         ),
#         "Topmost face",
#         "festival",
#     ),
#     Benchmark(
#         Intersection(
#             [IsObject("Face"), Complement(Map(IsObject("Guitar"), IsObject("Face"), GetAbove()))]
#         ),
#         "Faces of people not playing guitar",
#         "festival",
#     ),
#     Benchmark(
#         Intersection([Complement(IsSmiling()), Map(IsObject("Face"), IsObject("Face"), GetAbove())]),
#         "All faces in back that are not smiling and are behind another face",
#         "wedding",
#     ),
#     Benchmark(
#         Intersection([IsObject("Face"), Complement(Map(IsObject("Bride"), IsObject("Face"), GetContains()))]),
#         "All faces that are not the bride's face",
#         "wedding",
#     ),
#    Benchmark(
#         Union([EyesOpen(), Map(IsObject("Bride"), IsObject("Face"), GetContains())]),
#         "The bride's face and faces with eyes open",
#         "wedding",
#     ),
# ]

