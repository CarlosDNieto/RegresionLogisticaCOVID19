df <- read.csv("casos-asociados-a-covid-19.csv", header = T)

modelo <- glm(TIPO.PACIENTE ~ SEXO + EDAD + EMBARAZO + DIABETES + EPOC + ASMA +
                INMUNOSUPRESION + HIPERTENSION + OTRA.COMPLICACION + CARDIOVASCULAR +
                OBESIDAD + RENAL.CRONICA + TABAQUISMO, data = df, family = "binomial")

summary(modelo)
