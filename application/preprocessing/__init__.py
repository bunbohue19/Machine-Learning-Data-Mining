import itertools

from .methods import *
from inputs import *

# Don't ask me how I got those magic numbers in standardise methods...
# I got it from our metrics (see the magic notebook)


def preprocess_inputs(args: list) -> list[float]:
    preprocessed_inputs: list[list[float]] = [
        # Marital status
        one_hot(args[0], marital_status_inputs),
        # Application mode
        one_hot(args[1], application_mode_inputs),
        # Application order
        raw(float(args[2])),
        # Course
        one_hot(args[3], course_inputs),
        # Daytime / Evening attendance
        get_raw(args[4], daytime_evening_attendance_inputs),
        # Previous qualification
        one_hot(args[5], previous_qualification_inputs),
        # Previous qualification grade
        standard(args[6], (132.92060606060494, 13.236549060750294)),
        # Nationality
        one_hot(args[7], nationality_inputs),
        # Mother's qualification
        one_hot(args[8], mothers_qualification_inputs),
        # Father's qualification
        one_hot(args[9], fathers_qualification_inputs),
        # Mother's occupation
        one_hot(args[10], mothers_occupation_inputs),
        # Father's occupation
        one_hot(args[11], fathers_occupation_inputs),
        # Admission grade
        standard(args[12], (127.29393939393924, 14.609282556743532)),
        # Displaced
        get_raw(args[13], displaced_inputs),
        # Educational special needs
        get_raw(args[14], educational_special_needs_inputs),
        # Debtor
        get_raw(args[15], debtor_inputs),
        # Tuition fees up to date
        get_raw(args[16], tuition_fees_up_to_date_inputs),
        # Gender
        get_raw(args[17], gender_inputs),
        # Scholarship holder
        get_raw(args[18], scholarship_holder_inputs),
        # Age at enrollment
        raw(args[19]),
        # International
        get_raw(args[20], international_inputs),
        # Curricular units 1st sem (credited)
        raw(args[21]),
        # Curricular units 1st sem (enrolled)
        raw(args[22]),
        # Curricular units 1st sem (evaluations)
        raw(args[23]),
        # Curricular units 1st sem (approved)
        raw(args[24]),
        # Curricular units 1st sem (grade)
        raw(args[25]),
        # Curricular units 1st sem (without evaluations)
        raw(args[26]),
        # Curricular units 2nd sem (credited)
        raw(args[27]),
        # Curricular units 2nd sem (enrolled)
        raw(args[28]),
        # Curricular units 2nd sem (evaluations)
        raw(args[29]),
        # Curricular units 2nd sem (approved)
        raw(args[30]),
        # Curricular units 2nd sem (grade)
        raw(args[31]),
        # Curricular units 2nd sem (without evaluations)
        raw(args[32]),
        # Unemployment rate
        standard(args[33], (11.63035812672188, 2.667284425747152)),
        # Inflation rate
        standard(args[34], (1.2315977961432394, 1.3847204104409045)),
        # GDP
        standard(args[35], (-0.009256198347107725, 2.259674553698473))
    ]

    return list(itertools.chain.from_iterable(preprocessed_inputs))
