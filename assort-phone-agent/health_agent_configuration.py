from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from dataclasses import field

@dataclass
class PatientInfo:
    # - Collect patient's name and date of birth
    name: Optional[str] = None
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
    available_providers: List[Tuple[str, List[datetime], Dict[str, str]]] = field(default_factory=lambda: [
        (
            "Dr. Emily Carter",
            [datetime(2023, 5, 1, 10, 0), datetime(2023, 5, 2, 14, 30)],
            {
                "background": "Board-certified in Internal Medicine with 15 years of experience. Graduated from Harvard Medical School and completed residency at Massachusetts General Hospital. Special interest in preventive medicine and women's health.",
                "specialty": "Primary Care"
            }
        ),
        (
            "Dr. Michael Chen",
            [datetime(2023, 5, 3, 11, 15), datetime(2023, 5, 4, 9, 0)],
            {
                "background": "Fellowship-trained in Cardiology from Johns Hopkins. Published researcher in heart failure treatments. Expertise in non-invasive cardiac imaging and preventive cardiology. Board member of the American Heart Association.",
                "specialty": "Cardiology"
            }
        ),
        (
            "Dr. Sophia Rodriguez",
            [datetime(2023, 5, 4, 10, 0), datetime(2023, 5, 5, 14, 30)],
            {
                "background": "Board-certified in Pediatrics with a focus on adolescent medicine. Completed fellowship in Adolescent Medicine at Children's Hospital of Philadelphia. Advocate for mental health awareness in teenagers.",
                "specialty": "Pediatrics"
            }
        ),
        (
            "Dr. James Wilson",
            [datetime(2023, 5, 2, 11, 30), datetime(2023, 5, 3, 15, 0)],
            {
                "background": "Double board-certified in Internal Medicine and Endocrinology. Specializes in diabetes management and thyroid disorders. Pioneered a telemedicine program for rural diabetes patients. Regular speaker at American Diabetes Association conferences.",
                "specialty": "Endocrinology"
            }
        ),
        (
            "Dr. Aisha Patel",
            [datetime(2023, 5, 1, 13, 0), datetime(2023, 5, 5, 10, 30)],
            {
                "background": "Board-certified in Family Medicine with additional certification in Sports Medicine. Former team physician for a professional soccer team. Expertise in non-surgical orthopedics and exercise prescription for chronic diseases.",
                "specialty": "Family Medicine & Sports Medicine"
            }
        ),
        (
            "Dr. Robert Nguyen",
            [datetime(2023, 5, 2, 9, 0), datetime(2023, 5, 4, 14, 0)],
            {
                "background": "Board-certified in Neurology with fellowship training in Movement Disorders. Conducts clinical trials on new treatments for Parkinson's disease. Developed a multidisciplinary approach to treating essential tremor. Fluent in English and Vietnamese.",
                "specialty": "Neurology"
            }
        )
    ])
    
class SchedulerAgent:
    def __init__(self):
        self.patient_info = PatientInfo()
        self.available_providers = AvailableProviders()
        
    def get_missing_info(self):
        return [field for field in self.patient_info.__annotations__ if getattr(self.patient_info, field) is None]
    
if __name__ == "__main__":
    agent = SchedulerAgent()
    print(agent.get_missing_info())