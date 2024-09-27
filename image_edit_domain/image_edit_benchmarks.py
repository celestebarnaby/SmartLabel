from image_edit_domain.image_edit_dsl import *
from benchmark import Benchmark


image_edit_benchmarks = [
    Benchmark(
        Intersection([IsSmiling(), Complement(MouthOpen())]),
        "All faces that are smiling and do not have mouth open",
        "wedding",
    ),
    Benchmark(
        Map(IsObject("Guitar"), IsObject("Face"), GetAbove()),
        "Faces playing guitar",
        "festival"
    ),
    Benchmark(
        Intersection([Complement(EyesOpen()), Map(IsObject("Guitar"), IsObject("Face"), GetAbove())]),
        "Faces with eyes closed playing guitar",
        "festival"
    ),
    Benchmark(
        Map(IsObject("Car"), Intersection([IsObject("Face"), Complement(IsSmiling())]), GetContains()),
        "People who aren't smiling inside cars",
        "cars"
    ),
    Benchmark(
       Union([IsObject("Bicycle"), Map(IsObject("Face"), IsObject("Car"), GetIsContained())]),
        "Bicycles and cars with people in them",
        "cars"
    ),
    Benchmark(
        Union([IsObject("Person"), IsObject("Bicycle"), IsObject("Car")]),
        "All bicycles, cars, and people",
        "cars",
    ),
    Benchmark(
        Intersection([IsSmiling(), EyesOpen(), Complement(MouthOpen())]),
        "Faces that are smiling, have eyes open, and mouth closed",
        "wedding",
    ),

    Benchmark(
        Intersection([IsObject("Face"), Complement(IsSmiling()), Complement(IsObject("Bride"))]),
        "All faces that are not smiling and are not the bride's face",
        "wedding",
    ),
     Benchmark(
        Union([Map(IsObject("Suit"), IsObject("Face"), GetAbove()), IsObject("Bride"), IsObject("Wedding Gown")]),
        "Faces of people wearing suits, the bride, and the wedding gown",
        "wedding",
    ),
    Benchmark(
        Map(Map(IsObject("Face"), IsObject("Face"), GetRight()), IsObject("Face"), GetRight()),
        "All faces except 2 leftmost faces",
        "wedding",
    ),
    Benchmark(
        Union([IsObject("Guitar"), Map(IsObject("Guitar"), IsObject("Face"), GetAbove())]),
        "Guitars and people playing guitar",
        "festival",
    ),
    Benchmark(
        Map(Union([IsObject("Guitar"), IsSmiling()]), IsSmiling(), GetLeft()),
        "Smiling faces to the left of guitars or other smiling faces",
        "festival"
    ),
    Benchmark(
        Map(Union([IsObject("Guitar"), IsObject("Microphone"), IsObject("Speaker")]), IsSmiling(), GetAbove()),
        "Smiling faces to the left of guitars or microphones",
        "festival"
    ),
    Benchmark(
        Union([IsObject("Bicycle"), Map(IsObject("Helmet"), IsObject("Person"), GetBelow())]),
        "Bicycles and people wearing helmets",
        "cars",
    ),
    Benchmark(
        Map(Map(IsObject("Helmet"), IsObject("Person"), GetBelow()), IsObject("Bicycle"), GetBelow()),
        "Bicycles ridden by people wearing helmets",
        "cars",
    ),
    Benchmark(
        Map(Map(IsObject("Car"), IsObject("Bicycle"), GetRight()), IsObject("Person"), GetAbove()),
        "People on bicycles next to cars",
        "cars",
    ),
    Benchmark(
        Map(Union([IsObject("Bicycle"), IsObject("Car"), IsObject("Bus")]), IsObject("Person"), GetRight()),
        "People to the right of bicycles, cars, or buses",
        "cars",
    ),
    Benchmark(
        Intersection(
            [IsObject("Person"), Complement(Map(IsObject("Bicycle"), IsObject("Person"), GetAbove()))]
        ),
        "People not riding bicycles",
        "cars",
    ),
    Benchmark(
        Intersection(
            [IsObject("Face"), Complement(Map(IsObject("Car"), IsObject("Face"), GetContains()))]
        ),
        "People not in cars",
        "cars",
    ),
    Benchmark(
        Intersection(
            [IsObject("Car"), Complement(Map(IsObject("Face"), IsObject("Car"), GetIsContained()))]
        ),
        "Cars without people in them",
        "cars",
    ),
    Benchmark(
        Intersection(
            [
                IsObject("Face"),
                Complement(Map(IsObject("Face"), IsObject("Face"), GetBelow())),
            ]
        ),
        "Topmost face",
        "festival",
    ),
    Benchmark(
        Intersection(
            [IsObject("Face"), Complement(Map(IsObject("Guitar"), IsObject("Face"), GetAbove()))]
        ),
        "Faces of people not playing guitar",
        "festival",
    ),
    Benchmark(
        Intersection([Complement(IsSmiling()), Map(IsObject("Face"), IsObject("Face"), GetAbove())]),
        "All faces in back that are not smiling and are behind another face",
        "wedding",
    ),
    Benchmark(
        Intersection([IsObject("Face"), Complement(Map(IsObject("Bride"), IsObject("Face"), GetContains()))]),
        "All faces that are not the bride's face",
        "wedding",
    ),
   Benchmark(
        Union([EyesOpen(), Map(IsObject("Bride"), IsObject("Face"), GetContains())]),
        "The bride's face and faces with eyes open",
        "wedding",
    ),
]

