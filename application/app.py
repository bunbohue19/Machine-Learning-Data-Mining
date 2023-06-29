import gradio

# Don't know why relative import like below doesn't work
# from . import inputs

# Only this one works
from inputs import *
import predict
from models import *

with gradio.Blocks() as demo:
    gradio.Markdown("""
    # Predict students' dropout and academic success

    Dataset used from https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success,
    licensed under [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode)
    """)

    with gradio.Row():
        with gradio.Column() as input_column:
            marital_status = gradio.Dropdown(
                label="Marital status",
                choices=get_entries_names_list(marital_status_inputs),
            )
            application_mode = gradio.Dropdown(
                label="Application mode",
                choices=get_entries_names_list(application_mode_inputs),
            )
            application_order = gradio.Slider(
                label="Application Order",
                minimum=0,
                maximum=9,
                step=1,
            )
            course = gradio.Dropdown(
                label="Course",
                choices=get_entries_names_list(course_inputs),
            )
            daytime_attendance = gradio.Dropdown(
                label="Daytime / Evening attendance",
                choices=get_entries_names_list(
                    daytime_evening_attendance_inputs))
            prev_qualification = gradio.Dropdown(
                label="Previous qualification",
                choices=get_entries_names_list(previous_qualification_inputs))
            prev_quali_grade = gradio.Number(
                label="Previous qualification (Grade) (0 - 200)",
                minimum=0,
                maximum=200,
            )
            nationality = gradio.Dropdown(
                label="Nationality",
                choices=get_entries_names_list(nationality_inputs))
            mothers_quali = gradio.Dropdown(
                label="Mother's qualification",
                choices=get_entries_names_list(mothers_qualification_inputs))
            fathers_quali = gradio.Dropdown(
                label="Father's qualification",
                choices=get_entries_names_list(fathers_qualification_inputs))
            mothers_occu = gradio.Dropdown(
                label="Mother's occupation",
                choices=get_entries_names_list(mothers_occupation_inputs))
            fathers_occu = gradio.Dropdown(
                label="Father's occupation",
                choices=get_entries_names_list(fathers_occupation_inputs))
            addimision_grade = gradio.Number(
                label="Addimision grade (0 - 200)",
                minimum=0,
                maximum=200,
            )
            displaced = gradio.Dropdown(
                label="Displaced",
                choices=get_entries_names_list(displaced_inputs))
            educational_special_needs = gradio.Dropdown(
                label="Educational special needs",
                choices=get_entries_names_list(
                    educational_special_needs_inputs))
            debtor = gradio.Dropdown(
                label="Debtor", choices=get_entries_names_list(debtor_inputs))
            tuition_fee = gradio.Dropdown(
                label="Tuition fees up to date",
                choices=get_entries_names_list(tuition_fees_up_to_date_inputs))
            gender = gradio.Dropdown(
                label="Gender", choices=get_entries_names_list(gender_inputs))
            scholarship_holder = gradio.Dropdown(
                label="Scholarship holder",
                choices=get_entries_names_list(scholarship_holder_inputs))
            enrollment_age = gradio.Number(
                label="Age at enrollment",
                minimum=0,
            )
            international = gradio.Dropdown(
                label="International",
                choices=get_entries_names_list(international_inputs))
            units_1st_sem_credit = gradio.Number(
                label="Curricular units 1st sem (credited)",
                minimum=0,
            )
            units_1st_sem_enroll = gradio.Number(
                label="Curricular units 1st sem (enrolled)",
                minimum=0,
            )
            units_1st_sem_evals = gradio.Number(
                label="Curricular units 1st sem (evaluations)",
                minimum=0,
            )
            units_1st_sem_approved = gradio.Number(
                label="Curricular units 1st sem (approved)",
                minimum=0,
            )
            units_1st_sem_grade = gradio.Number(
                label="Curricular units 1st sem (grade) (0 - 20)",
                minimum=0,
                maximum=20,
            )
            units_1st_sem_noeval = gradio.Number(
                label="Curricular units 1st sem (without evaluations)",
                minimum=0,
            )
            units_2nd_sem_credit = gradio.Number(
                label="Curricular units 2nd sem (credited)",
                minimum=0,
            )
            units_2nd_sem_enroll = gradio.Number(
                label="Curricular units 2nd sem (enrolled)",
                minimum=0,
            )
            units_2nd_sem_evals = gradio.Number(
                label="Curricular units 2nd sem (evaluations)",
                minimum=0,
            )
            units_2nd_sem_approved = gradio.Number(
                label="Curricular units 2nd sem (approved)",
                minimum=0,
            )
            units_2nd_sem_grade = gradio.Number(
                label="Curricular units 2nd sem (grade) (0 - 20)",
                minimum=0,
                maximum=20,
            )
            units_2nd_sem_noeval = gradio.Number(
                label="Curricular units 2nd sem (without evaluations)",
                minimum=0,
            )
            unemploy_rate = gradio.Number(label="Unemployment rate (%)",
                                          minimum=0,
                                          maximum=100)
            inflation_rate = gradio.Number(label="Inflation rate (%)",
                                           minimum=0,
                                           maximum=100)
            gdp = gradio.Number(label="GDP")
            model_field = gradio.Dropdown(
                label="Model",
                choices=get_entries_names_list(model_choice_inputs),
            )

        with gradio.Column() as output_column:
            output = gradio.Textbox(label="Output / Evaluation")
            with gradio.Row() as btn_row:
                predict_btn = gradio.Button(value="Predict")
                predict_btn.click(
                    fn=predict.predict_fn,
                    inputs=[
                        marital_status,
                        application_mode,
                        application_order,
                        course,
                        daytime_attendance,
                        prev_qualification,
                        prev_quali_grade,
                        nationality,
                        mothers_quali,
                        fathers_quali,
                        mothers_occu,
                        fathers_occu,
                        addimision_grade,
                        displaced,
                        educational_special_needs,
                        debtor,
                        tuition_fee,
                        gender,
                        scholarship_holder,
                        enrollment_age,
                        international,
                        units_1st_sem_credit,
                        units_1st_sem_enroll,
                        units_1st_sem_evals,
                        units_1st_sem_approved,
                        units_1st_sem_grade,
                        units_1st_sem_noeval,
                        units_2nd_sem_credit,
                        units_2nd_sem_enroll,
                        units_2nd_sem_evals,
                        units_2nd_sem_approved,
                        units_2nd_sem_grade,
                        units_2nd_sem_noeval,
                        unemploy_rate,
                        inflation_rate,
                        gdp,
                        model_field,
                    ],
                    outputs=[output],
                )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
