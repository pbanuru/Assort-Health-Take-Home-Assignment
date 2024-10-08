from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from dataclasses import field
from livekit.agents import (
    llm,
    JobContext,
)
from livekit.agents.pipeline import VoicePipelineAgent
from typing import Annotated
from colorama import Fore, Style
import os
from livekit import api
from send_email import send_email

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

    confirmation_email_sent: Optional[bool] = False
    information_confirmed: Optional[bool] = False  # New flag


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
    def __init__(self, job_context: JobContext):
        super().__init__()
        self.patient_info = PatientInfo()
        self.available_providers = AvailableProviders()
        self.job_context = job_context
        self.livekit_client = api.LiveKitAPI(
            os.getenv("LIVEKIT_URL"),
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET"),
        )

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
   - Reason for the visit (Chief medical complaint) (don't refer to it as the chief complaint cause patient may not know what that is)
   - Address (collect street address, city, state, and ZIP code separately)
   - Contact information - phone number and email (mention that email is optional but required for receiving a confirmation email)

3. Based on the patient's chief complaint, suggest appropriate available providers and appointment times.
4. Help the patient select a provider and appointment time. Always mention the provider's name before their other information.
5. Confirm all collected information with the patient.
6. If an email was provided, inform the patient that a confirmation email will be sent to their email with their appointment details. If no email was provided, inform them that they won't receive a confirmation email.

Important notes:
- If any information was not clearly heard, apologize and ask the patient to spell it out. 
- Repeat the information back to the patient both during (if not clear) and after collection.
- Ensure all required information is collected before concluding the call.
- Be patient, professional, and empathetic throughout the conversation.
- If the patient is unsure about any information, offer to skip it temporarily and return to it later.
- Use the available provider information to match the patient with an appropriate doctor based on their needs.
- If the patient does not have a referral, it's ok if the referred physician field is left blank.
- Before you hang up, make sure to check if all the necessary information has been gathered and the confirmation email has been sent (if applicable), and if you are ready to hang up.
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
        chat_ctx.messages[
            0
        ].content = f"""
        {self.get_system_prompt()}
        
        Missing information: {self.get_missing_info()}
        
        Gathered information: {self.get_gathered_info()}
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
        return f"Patient's first name set to {first_name}. Please continue the conversation."

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
        return (
            f"Patient's last name set to {last_name}. Please continue the conversation."
        )

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
            return f"Patient's date of birth set to {date_of_birth}. Please continue the conversation."
        except ValueError:
            return "Invalid date format provided for date of birth. Please try again and continue the conversation."

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
        return f"Patient's insurance provider set to {insurance_payer}. Please continue the conversation."

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
        return f"Patient's insurance ID set to {insurance_id}. Please continue the conversation."

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
        return f"Patient's referral status set to: {'has referral' if has_referral else 'no referral'}. Please continue the conversation."

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
        return f"Referring physician set to Dr. {referred_physician}. Please continue the conversation."

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
        return f"Patient's main reason for visit set to: {chief_complaint}. Please continue the conversation."

    @llm.ai_callable()
    async def set_address(
        self,
        street_address: Annotated[
            str, llm.TypeInfo(description="The street address of the patient")
        ],
        city: Annotated[
            str, llm.TypeInfo(description="The city of the patient's address")
        ] = None,
        state: Annotated[
            str, llm.TypeInfo(description="The state of the patient's address")
        ] = None,
        zip_code: Annotated[
            str, llm.TypeInfo(description="The ZIP code of the patient's address")
        ] = None,
    ):
        """Called when the user provides their address."""
        missing_info = []
        if not city:
            missing_info.append("city")
        if not state:
            missing_info.append("state")
        if not zip_code:
            missing_info.append("ZIP code")

        if missing_info:
            missing_fields = ", ".join(missing_info)
            return f"Street address set to {street_address}. Missing address information: {missing_fields}. Please continue the conversation."

        full_address = f"{street_address}, {city}, {state} {zip_code}"
        self.patient_info.address = full_address
        if DEBUG:
            print(Fore.RED + f"Address: {full_address}" + Style.RESET_ALL)
        return f"Patient's full address set to: {full_address}. Please continue the conversation."

    @llm.ai_callable()
    async def set_phone_number(
        self,
        phone_number: Annotated[
            str, llm.TypeInfo(description="The patient's phone number")
        ],
    ):
        """Called when the user provides their phone number."""
        self.patient_info.phone_number = phone_number
        if DEBUG:
            print(Fore.RED + f"Phone number: {phone_number}" + Style.RESET_ALL)
        return f"Patient's phone number set to: {phone_number}. Please continue the conversation."

    @llm.ai_callable()
    async def set_email(
        self,
        email: Annotated[
            str, llm.TypeInfo(description="The patient's email address (optional)")
        ],
    ):
        """Called when the user provides their email address."""
        self.patient_info.email = email
        if DEBUG:
            print(Fore.RED + f"Email: {email}" + Style.RESET_ALL)
        return f"Patient's email address set to: {email}. Confirmation email will be sent to this address. Please continue the conversation."

    @llm.ai_callable()
    async def check_all_info_gathered(self):
        """Check if all necessary patient information has been gathered (before sending confirmation email)."""
        missing_info = self.get_missing_info()
        missing_info.pop("confirmation_email_sent", None)
        missing_info.pop("email", None)  # Email is now optional

        if not self.patient_info.has_referral:
            missing_info.pop("referred_physician", None)

        if DEBUG:
            print(Fore.GREEN + f"Missing info: {missing_info}" + Style.RESET_ALL)
            print(
                Fore.GREEN
                + f"Gathered info: {self.get_gathered_info()}"
                + Style.RESET_ALL
            )

        if not missing_info:
            return "All necessary information gathered. Ready to proceed with appointment scheduling. Please continue the conversation."
        else:
            missing_fields = ", ".join(missing_info.keys())
            return f"Information still needed: {missing_fields}. Please continue the conversation."

    @llm.ai_callable()
    async def confirmation_completed_signal(self):
        """Called when all the gathered information has been confirmed with the patient near the end of the call. Needs to be called before sending a confirmation email."""
        if not self.patient_info.information_confirmed:
            self.patient_info.information_confirmed = True
            if DEBUG:
                print(
                    Fore.GREEN
                    + "Information confirmed with the patient."
                    + Style.RESET_ALL
                )
            return "Information has been confirmed with the patient. Please continue the conversation."
        else:
            if DEBUG:
                print(
                    Fore.YELLOW + "Information was already confirmed." + Style.RESET_ALL
                )
            return (
                "Information was already confirmed. Please continue the conversation."
            )

    def _can_send_email(self):
        """Check if email can be sent."""
        if self.patient_info.email is None:
            if DEBUG:
                print(
                    Fore.YELLOW
                    + "No email address provided. Skipping confirmation email."
                    + Style.RESET_ALL
                )
            return False, "No email address provided. Confirmation email not sent."

        if not self.patient_info.information_confirmed:
            if DEBUG:
                print(
                    Fore.YELLOW
                    + "Information not confirmed. Skipping confirmation email."
                    + Style.RESET_ALL
                )
            return False, "Information not confirmed. Confirmation email not sent."

        return True, ""

    def _create_email_body(self):
        """Create the email body."""
        return f"""
Dear {self.patient_info.first_name} {self.patient_info.last_name},

Thank you for scheduling an appointment with Assort Health. Your appointment details are as follows:

Provider: {self.patient_info.upcoming_appointment_provider}
Date and Time: {self.patient_info.upcoming_appointment_time.strftime('%A, %B %d, %Y at %I:%M %p')}

Please arrive 15 minutes before your scheduled appointment time. If you need to reschedule or have any questions, please call our office.

We look forward to seeing you soon.

Best regards,
Assort Health Team
        """

    def _handle_email_result(self, success):
        """Handle the result of sending the email."""
        if success:
            self.patient_info.confirmation_email_sent = True
            if DEBUG:
                print(
                    Fore.GREEN
                    + f"Confirmation email sent to: {self.patient_info.email}"
                    + Style.RESET_ALL
                )
            return f"Confirmation email sent to {self.patient_info.email}. Please inform the patient and continue the conversation."
        else:
            if DEBUG:
                print(
                    Fore.RED
                    + f"Failed to send confirmation email to: {self.patient_info.email}"
                    + Style.RESET_ALL
                )
            return f"Failed to send confirmation email to {self.patient_info.email}. Please inform the patient and continue the conversation."

    @llm.ai_callable()
    async def send_confirmation_email(self):
        """Send a confirmation email to the patient with their appointment details. Needs to be called AFTER confirmation_completed_signal."""
        can_send, message = self._can_send_email()
        if not can_send:
            return message + " Please continue the conversation."

        subject = "Appointment Confirmation - Assort Health"
        body = self._create_email_body()

        email = "pbanuru10@gmail.com" if DEBUG else "jeff@assorthealth.com"
        success = send_email(email, subject, body)
        return self._handle_email_result(success)

    @llm.ai_callable()
    async def hang_up(self):
        """Check if all necessary information has been gathered, confirmed, and that the confirmation email has been sent if applicable, and hang up the call."""
        missing_info = self.get_missing_info()
        missing_info.pop("confirmation_email_sent", None)
        missing_info.pop("email", None)  # Email is now optional

        if not self.patient_info.has_referral:
            missing_info.pop("referred_physician", None)

        if DEBUG:
            print(
                Fore.YELLOW
                + f"Hang up check - Missing info: {missing_info}"
                + Style.RESET_ALL
            )
            print(
                Fore.YELLOW
                + f"Email provided: {self.patient_info.email is not None}"
                + Style.RESET_ALL
            )
            print(
                Fore.YELLOW
                + f"Information confirmed: {self.patient_info.information_confirmed}"
                + Style.RESET_ALL
            )
            print(
                Fore.YELLOW
                + f"Confirmation email sent: {self.patient_info.confirmation_email_sent}"
                + Style.RESET_ALL
            )

        if not missing_info and self.patient_info.information_confirmed:
            if self.patient_info.email is None:
                if DEBUG:
                    print(
                        Fore.GREEN
                        + "Ready to hang up: All information gathered and confirmed, no email provided"
                        + Style.RESET_ALL
                    )
                await self.delete_room()
                return "All necessary information gathered and confirmed. No email provided. Call concluded and room deleted."
            elif not self.patient_info.confirmation_email_sent:
                return "All necessary information gathered and confirmed. Confirmation email needs to be sent before concluding the call. Please continue the conversation."
            else:
                if DEBUG:
                    print(
                        Fore.GREEN
                        + "Ready to hang up: All information gathered, confirmed, and email sent"
                        + Style.RESET_ALL
                    )
                await self.delete_room()
                return "All necessary information gathered, confirmed, and confirmation email sent. Call concluded and room deleted."
        else:
            if not self.patient_info.information_confirmed:
                return "Unable to conclude call. Information has not been confirmed with the patient. Please continue the conversation."
            missing_fields = ", ".join(missing_info.keys())
            return f"Unable to conclude call. Missing information: {missing_fields}. Please continue the conversation."

    async def delete_room(self):
        """Delete the LiveKit room after the call is concluded."""
        try:
            await self.livekit_client.room.delete_room(
                api.DeleteRoomRequest(room=self.job_context.room.name)
            )
            if DEBUG:
                print(
                    Fore.GREEN
                    + f"Room {self.job_context.room.name} deleted successfully"
                    + Style.RESET_ALL
                )
        except Exception as e:
            if DEBUG:
                print(Fore.RED + f"Error deleting room: {str(e)}" + Style.RESET_ALL)

    @llm.ai_callable()
    async def set_appointment(
        self,
        provider: Annotated[
            str,
            llm.TypeInfo(description="The name of the selected healthcare provider"),
        ],
        appointment_time: Annotated[
            str,
            llm.TypeInfo(
                description="The selected appointment time (YYYY-MM-DD HH:MM)"
            ),
        ],
    ):
        """Called when the user selects a provider and appointment time."""
        self.patient_info.upcoming_appointment_provider = provider
        try:
            self.patient_info.upcoming_appointment_time = datetime.strptime(
                appointment_time, "%Y-%m-%d %H:%M"
            )
            if DEBUG:
                print(
                    Fore.RED
                    + f"Appointment set with {provider} at {appointment_time}"
                    + Style.RESET_ALL
                )
            return f"Appointment scheduled with {provider} on {appointment_time}. Please continue the conversation and confirm the appointment details with the patient."
        except ValueError:
            return "Invalid date and time format provided for appointment. Please try again and continue the conversation."

    @llm.ai_callable()
    async def suggest_providers(self):
        """Suggest appropriate providers based on the patient's chief complaint."""
        if self.patient_info.chief_complaint is None:
            return "Chief complaint not provided. Unable to suggest providers. Please ask the patient for their chief complaint and continue the conversation."

        providers_info = []
        for provider, times, info in self.available_providers.available_providers:
            provider_data = {
                "name": provider,
                "specialty": info["specialty"],
                "background": info["background"],
                "available_times": [t.strftime("%A, %B %d at %I:%M %p") for t in times],
            }
            providers_info.append(provider_data)

        response = f"""Chief complaint: "{self.patient_info.chief_complaint}"
        Available providers:

        """
        for provider in providers_info:
            response += f"""
        Name: {provider['name']}
        Specialty: {provider['specialty']}
        Background: {provider['background']}
        Available times: {', '.join(provider['available_times'])}

        """

        response += """
        Please suggest up to three appropriate providers based on the chief complaint. For each suggested provider, include their name, specialty, available times, and a brief explanation of why they would be a good fit for the patient's needs. If no providers are a good fit, suggest creating a new provider. After suggesting providers, please continue the conversation with the patient to help them choose a provider and appointment time."""

        return response


if __name__ == "__main__":
    agent = SchedulerAgent()
    print(agent.get_missing_info())
    # You can test the new function here if needed
    # import asyncio
    # print(asyncio.run(agent.hang_up()))
