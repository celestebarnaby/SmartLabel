from image_search_domain.image_search_dsl import *
from benchmark import Benchmark

image_search_benchmarks = [

    # PhotoScout tasks
    Benchmark(
        Map(IsObject("Car"), IsObject("Bicycle"), GetAround()),
        "PhotoScout task 0",
        "cars"
    ),
    Benchmark(
        Map(IsObject("Guitar"), IsObject("Microphone"), GetAround()),
        "PhotoScout task 1",
        "festival",
    ),
    Benchmark(
        Union([IsObject("Face"), IsObject("Person")]),
        "PhotoScout task 2",
        "festival",
    ),
    Benchmark(
        Map(IsObject("Suit"), IsObject("Bride") , GetLeft()),
        "PhotoScout task 3 (modified - bride is to the left of a person wearing a suit)",
        "wedding",
    ),
    Benchmark(
        Map(Map(IsObject("Bride"), IsObject("Face"), GetAbove()), IsObject("Suit"), GetNotAround()),
        "PhotoScout task 4",
        "wedding",
    ),

    # Visual concept tasks
    Benchmark(
        Map(IsObject("Face"), IsObject("Glasses"), GetContains()),
        "People wearing glasses",
        "wedding",
    ),
    Benchmark(
        Intersection([IsObject("Person"), Complement(Map(IsObject("Glasses"), IsObject("Person"), GetIsContained()))]),
        "People without glasses",
        "wedding",
    ),
    Benchmark(
        Map(MouthOpen(), MouthOpen(), GetRight()),
        "People talking",
        "wedding",
    ),
    Benchmark(
        Intersection([IsSmiling(), Map(Intersection([MouthOpen(), IsSmiling()]), MouthOpen(), GetRight())]),
        "People laughing",
        "wedding",
    ),
    Benchmark(
        Map(IsObject("Tie"), IsObject("Face"), GetAbove()),
        "People in ties",
        "wedding",
    ),
    Benchmark(
        Union([IsObject("Suit"), IsObject("Tie"), IsObject("Wedding Gown")]),
        "Formal attire",
        "wedding",
    ),
    Benchmark(
        Map(IsObject("Bride"), IsSmiling(), GetAbove()),
        "A happy bride",
        "wedding",
    ),
    Benchmark(
        Map(IsObject("Guitar"), IsObject("Face"), GetAbove()),
        "Guitar players",
        "festival",
    ),
    Benchmark(
        Map(IsObject("Microphone"), IsObject("Face"), GetAbove()),
        "Singers",
        "festival",
    ),
    Benchmark(
        Map(IsObject("Guitar"), Map(IsObject("Microphone"), IsObject("Face"), GetAbove()), GetAbove()),
        "People singing and playing guitar",
        "festival",
    ),
    Benchmark(
        Map(IsObject("Microphone"), IsSmiling(), GetAround()),
        "Happy performers",
        "festival",
    ),
    Benchmark(
        Union([IsObject("Speaker"), IsObject("Microphone"), IsObject("Guitar")]),
        "Stage equipment",
        "festival",
    ),
    Benchmark(
        Map(IsObject("Person"), IsObject("Bicycle"), GetBelow()),
        "Cyclists",
        "cars",
    ),
    Benchmark(
        Map(IsObject("Bicycle"), IsObject("Car"), GetNotAround()),
        "Carfree streets",
        "cars",
    ),
    Benchmark(
        Map(IsSmiling(), IsObject("Bicycle"), GetBelow()),
        "Happy cyclists",
        "cars",
    ),
    Benchmark(
        Map(Map(IsObject("Helmet"), IsObject("Person"), GetBelow()), IsObject("Bicycle"), GetBelow()),
        "Cyclists with helmets",
        "cars",
    ),
    Benchmark(
        Map(IsObject("Car"), IsObject("Face"), GetContains()),
        "People driving cars",
        "cars",
    ),
    Benchmark(
        Intersection(
            [IsObject("Face"), Complement(Map(IsObject("Car"), IsObject("Face"), GetContains()))]
        ),
        "Pedestrians and cyclists",
        "cars",
    ),
    Benchmark(
        Union([IsObject("Bicycle"), IsObject("Car")]),
        "Modes of transportation",
        "cars",
    ),
    Benchmark(
        Map(IsObject("Hat"), IsObject("Face"), GetBelow()),
        "People wearing hats",
        "cars",
    ),
]