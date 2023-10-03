
class preprocMedicalAppointment:

    def __init__(self) -> None:
        pass

    def remove_outliers_age(self, df_medical_appointment, keep_range=(0, 100)):
        """
         Remove records whose age is outside the keep_range.
         
         @param df_medical_appointment - Data frame of appointments
         @param keep_range - Range to keep ( min max )
         
         @return Data frame of the same type as df_medical
        """
        
        return df_medical_appointment.loc[
            (df_medical_appointment['Age'] >= keep_range[0])
            & (df_medical_appointment['Age'] <= keep_range[1])
            ]