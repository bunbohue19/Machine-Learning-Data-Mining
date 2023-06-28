from .typings import DropdownValuesList

from .application_mode_inputs import *
from .course_inputs import *
from .daytime_evening_attendance_inputs import *
from .debtor_inputs import *
from .displaced_inputs import *
from .educational_special_needs_inputs import *
from .fathers_occupation_inputs import *
from .fathers_qualification_inputs import *
from .gender_inputs import *
from .international_inputs import *
from .marital_status_inputs import *
from .model_choice_inputs import *
from .mothers_occupation_inputs import *
from .mothers_qualification_inputs import *
from .nationality_inputs import *
from .previous_qualification_inputs import *
from .scholarship_holder_inputs import *
from .tuition_fees_up_to_date_inputs import *


def get_entries_names_list(entries: DropdownValuesList) -> list[str]:
    return [key for (key, _) in entries]
