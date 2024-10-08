from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from dataclasses import field
from livekit.agents import (
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from typing import Annotated
from colorama import Fore, Style

DEBUG = True


@dataclass
class PatientInfo:
    # - Collect patient's name and date of birth
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    # - Collect insurance information
    #     - Payer name and ID
    insurance_payer: Optional[str] = None
    insurance_id: Optional[str] = None
    # - Ask if they have a referral, and to which physician
    has_referral: Optional[bool] = None
    referred_physician: Optional[str] = None
    # - Collect chief medical complaint/reason they are coming in
    chief_complaint: Optional[str] = None
    # - Collect other demographics like address
    address: Optional[str] = None
    # - Collect contact information: phone number and optionally email
    phone_number: Optional[str] = None
    email: Optional[str] = None
    # - Offer up best available providers and times
    upcoming_appointment_provider: Optional[str] = None
    upcoming_appointment_time: Optional[datetime] = None


@dataclass
class AvailableProviders:
    # - Offer up best available providers and times
    #     - Using fake data with made up doctors
    available_providers: List[Tuple[str, List[datetime], Dict[str, str]]] = field(
        default_factory=lambda: [
            (
                "Dr. Emily Carter",
                [datetime(2023, 5, 1, 10, 0), datetime(2023, 5, 2, 14, 30)],
                {
                    "background": "Board-certified in Internal Medicine with 15 years of experience. Graduated from Harvard Medical School and completed residency at Massachusetts General Hospital. Special interest in preventive medicine and women's health.",
                    "specialty": "Primary Care",
                },
            ),
            (
                "Dr. Michael Chen",
                [datetime(2023, 5, 3, 11, 15), datetime(2023, 5, 4, 9, 0)],
                {
                    "background": "Fellowship-trained in Cardiology from Johns Hopkins. Published researcher in heart failure treatments. Expertise in non-invasive cardiac imaging and preventive cardiology. Board member of the American Heart Association.",
                    "specialty": "Cardiology",
                },
            ),
            (
                "Dr. Sophia Rodriguez",
                [datetime(2023, 5, 4, 10, 0), datetime(2023, 5, 5, 14, 30)],
                {
                    "background": "Board-certified in Pediatrics with a focus on adolescent medicine. Completed fellowship in Adolescent Medicine at Children's Hospital of Philadelphia. Advocate for mental health awareness in teenagers.",
                    "specialty": "Pediatrics",
                },
            ),
            (
                "Dr. James Wilson",
                [datetime(2023, 5, 2, 11, 30), datetime(2023, 5, 3, 15, 0)],
                {
                    "background": "Double board-certified in Internal Medicine and Endocrinology. Specializes in diabetes management and thyroid disorders. Pioneered a telemedicine program for rural diabetes patients. Regular speaker at American Diabetes Association conferences.",
                    "specialty": "Endocrinology",
                },
            ),
            (
                "Dr. Aisha Patel",
                [datetime(2023, 5, 1, 13, 0), datetime(2023, 5, 5, 10, 30)],
                {
                    "background": "Board-certified in Family Medicine with additional certification in Sports Medicine. Former team physician for a professional soccer team. Expertise in non-surgical orthopedics and exercise prescription for chronic diseases.",
                    "specialty": "Family Medicine & Sports Medicine",
                },
            ),
            (
                "Dr. Robert Nguyen",
                [datetime(2023, 5, 2, 9, 0), datetime(2023, 5, 4, 14, 0)],
                {
                    "background": "Board-certified in Neurology with fellowship training in Movement Disorders. Conducts clinical trials on new treatments for Parkinson's disease. Developed a multidisciplinary approach to treating essential tremor. Fluent in English and Vietnamese.",
                    "specialty": "Neurology",
                },
            ),
        ]
    )


class SchedulerAgent(llm.FunctionContext):
    def __init__(self):
        super().__init__()  # Call the parent class constructor
        self.patient_info = PatientInfo()
        self.available_providers = AvailableProviders()

    def get_system_prompt(self):
        return """
You are an AI medical appointment scheduler for Assort Health. Your task is to collect essential patient information and schedule an appointment with an appropriate healthcare provider. Follow these guidelines:

1. Greet the patient politely and explain your role.
2. Collect the following information:
   - Patient's first name (ask them to spell it out and repeat it back)
   - Patient's last name (ask them to spell it out and repeat it back)
   - Date of birth
   - Insurance information - payer name and ID
   - Referral status and referring physician (if applicable)
   - Reason for the visit (Chief medical complaint)
   - Address (collect street address, city, state, and ZIP code separately)
   - Contact information - phone number and email

3. Based on the patient's chief complaint, suggest appropriate available providers and appointment times.
4. Help the patient select a provider and appointment time.
5. Confirm all collected information with the patient.
6. Inform the patient that a confirmation email will be sent to their email with their appointment details.

Important notes:
- If any information was not clearly heard, apologize and ask the patient to spell it out. 
- Repeat the information back to the patient both during and after collection.
- Ensure all required information is collected before concluding the call.
- Be patient, professional, and empathetic throughout the conversation.
- If the patient is unsure about any information, offer to skip it temporarily and return to it later.
- Use the available provider information to match the patient with an appropriate doctor based on their needs.
- If the patient does not have a referral, it's ok if the referred physician field is left blank.

Remember to maintain a friendly and helpful demeanor throughout the interaction.
"""

    def get_missing_info(self):
        return {
            field: getattr(self.patient_info, field)
            for field in self.patient_info.__annotations__
            if getattr(self.patient_info, field) is None
        }

    def get_gathered_info(self):
        return {
            field: getattr(self.patient_info, field)
            for field in self.patient_info.__annotations__
            if getattr(self.patient_info, field) is not None
        }

    async def modify_before_llm(
        self, assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext
    ):
        provider_options = (
            ""
            if self.patient_info.upcoming_appointment_provider is not None
            else f"Available providers: {self.available_providers.available_providers}"
        )
        chat_ctx.messages[
            0
        ].content = f"""
        {self.get_system_prompt()}
        
        Missing information: {self.get_missing_info()}
        
        Gathered information: {self.get_gathered_info()}
        
        {provider_options}
        """
        # Remove the latest system message besides the initial system message
        for i in range(len(chat_ctx.messages) - 1, 0, -1):
            if chat_ctx.messages[i].role == "system":
                del chat_ctx.messages[i]
                break

        chat_ctx.messages.append(
            llm.ChatMessage(
                role="system",
                content=f"Missing information: {self.get_missing_info()} \n\nGathered information: {self.get_gathered_info()}",
            )
        )

        if DEBUG:
            print(Fore.GREEN + "START CONTEXT" + Style.RESET_ALL)
            print(chat_ctx.messages[0].content)
            print(Fore.GREEN + "END CONTEXT" + Style.RESET_ALL)

    @llm.ai_callable()
    async def set_first_name(
        self,
        first_name: Annotated[
            str, llm.TypeInfo(description="The first name of the patient")
        ],
    ):
        """Called when the user provides their first name."""
        self.patient_info.first_name = first_name
        if DEBUG:
            print(Fore.RED + f"First name: {first_name}" + Style.RESET_ALL)
        return f"{' '.join(first_name.split())} is that correct?"

    @llm.ai_callable()
    async def set_last_name(
        self,
        last_name: Annotated[
            str, llm.TypeInfo(description="The last name of the patient")
        ],
    ):
        """Called when the user provides their last name."""
        self.patient_info.last_name = last_name
        if DEBUG:
            print(Fore.RED + f"Last name: {last_name}" + Style.RESET_ALL)
        return f"{' '.join(last_name.split())} is that correct?"

    @llm.ai_callable()
    async def set_date_of_birth(
        self,
        date_of_birth: Annotated[
            str, llm.TypeInfo(description="The patient's date of birth (YYYY-MM-DD)")
        ],
    ):
        """Called when the user provides their date of birth."""
        try:
            self.patient_info.date_of_birth = datetime.strptime(
                date_of_birth, "%Y-%m-%d"
            )
            if DEBUG:
                print(Fore.RED + f"Date of birth: {date_of_birth}" + Style.RESET_ALL)
            return f"{date_of_birth} is that correct?"
        except ValueError:
            return "I'm sorry, that doesn't seem to be a valid date format. Please provide your date of birth in YYYY-MM-DD format."

    @llm.ai_callable()
    async def set_insurance_payer(
        self,
        insurance_payer: Annotated[
            str,
            llm.TypeInfo(description="The name of the patient's insurance provider"),
        ],
    ):
        """Called when the user provides their insurance payer information."""
        self.patient_info.insurance_payer = insurance_payer
        if DEBUG:
            print(Fore.RED + f"Insurance payer: {insurance_payer}" + Style.RESET_ALL)
        return f"I've recorded your insurance provider as {insurance_payer}. Is that correct?"

    @llm.ai_callable()
    async def set_insurance_id(
        self,
        insurance_id: Annotated[
            str, llm.TypeInfo(description="The patient's insurance ID number")
        ],
    ):
        """Called when the user provides their insurance ID."""
        self.patient_info.insurance_id = insurance_id
        if DEBUG:
            print(Fore.RED + f"Insurance ID: {insurance_id}" + Style.RESET_ALL)
        return f"I've recorded your insurance ID as {insurance_id}. Is that correct?"

    @llm.ai_callable()
    async def set_referral_status(
        self,
        has_referral: Annotated[
            bool, llm.TypeInfo(description="Whether the patient has a referral or not")
        ],
    ):
        """Called when the user indicates whether they have a referral or not."""
        self.patient_info.has_referral = has_referral
        if DEBUG:
            print(Fore.RED + f"Has referral: {has_referral}" + Style.RESET_ALL)
        return f"I've noted that you {'have' if has_referral else 'do not have'} a referral. Is that correct?"

    @llm.ai_callable()
    async def set_referred_physician(
        self,
        referred_physician: Annotated[
            str,
            llm.TypeInfo(
                description="The name of the physician who provided the referral"
            ),
        ],
    ):
        """Called when the user provides the name of the referring physician."""
        self.patient_info.referred_physician = referred_physician
        if DEBUG:
            print(
                Fore.RED + f"Referred physician: {referred_physician}" + Style.RESET_ALL
            )
        return f"I've recorded the referring physician as Dr. {referred_physician}. Is that correct?"

    @llm.ai_callable()
    async def set_chief_complaint(
        self,
        chief_complaint: Annotated[
            str,
            llm.TypeInfo(
                description="The patient's main reason for the visit or primary medical concern"
            ),
        ],
    ):
        """Called when the user provides their chief medical complaint or reason for the visit."""
        self.patient_info.chief_complaint = chief_complaint
        if DEBUG:
            print(Fore.RED + f"Chief complaint: {chief_complaint}" + Style.RESET_ALL)
        return f"I understand that your main reason for the visit is: {chief_complaint}. Is that correct?"

    @llm.ai_callable()
    async def set_address(
        self,
        street_address: Annotated[
            str, llm.TypeInfo(description="The street address of the patient")
        ],
        city: Annotated[
            str, llm.TypeInfo(description="The city of the patient's address")
        ],
        state: Annotated[
            str, llm.TypeInfo(description="The state of the patient's address")
        ],
        zip_code: Annotated[
            str, llm.TypeInfo(description="The ZIP code of the patient's address")
        ],
    ):
        """Called when the user provides their address."""
        full_address = f"{street_address}, {city}, {state} {zip_code}"
        self.patient_info.address = full_address
        if DEBUG:
            print(Fore.RED + f"Address: {full_address}" + Style.RESET_ALL)
        return f"I've recorded your address as: {full_address}. Is that correct?"


if __name__ == "__main__":
    agent = SchedulerAgent()
    print(agent.get_missing_info())
# Example chatcontext:
# ctx = llm.ChatContext(messages=[
#     ChatMessage(
#         role='system',
#         content='You are a medical appointment scheduler for Assort Health.',
#         id=None, name=None, tool_calls=None, tool_call_id=None, tool_exception=None
#     ),
#     ChatMessage(
#         role='assistant',
#         content='Hey, how can I help you today?',
#         id=None, name=None, tool_calls=None, tool_call_id=None, tool_exception=None
#     ),
#     ChatMessage(
#         role='user',
#         content='Hi. Who are you?',
#         id=None, name=None, tool_calls=None, tool_call_id=None, tool_exception=None
#     ),
#     ChatMessage(
#         role='assistant',
#         content="Hello! I'm a medical appointment scheduler for Assort Health. How can I assist you today?",
#         id=None, name=None, tool_calls=None, tool_call_id=None, tool_exception=None
#     ),
#     ChatMessage(
#         role='user',
#         content='Wow. So cool. What do you wanna talk about?',
#         id=None, name=None, tool_calls=None, tool_call_id=None, tool_exception=None
#     ),
#     ChatMessage(
#         role='assistant',
#         content="Thank you! I'm here to help with anything related to scheduling medical appointments, answering questions about our services, or assisting with any other healthcare-related needs you might have. Let me know how I can assist you!",
#         id=None, name=None, tool_calls=None, tool_call_id=None, tool_exception=None
#     ),
#     ChatMessage(
#         role='user',
#         content="That's really amazing.",
#         id=None, name=None, tool_calls=None, tool_call_id=None, tool_exception=None
#     )
# ])
