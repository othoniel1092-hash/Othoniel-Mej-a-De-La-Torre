# ==============================================================
# PROYECTO CRISP-DM: PREDICCION DE OZONO EN NUEVA YORK (1973)
# Dataset: airquality (paquete datasets de R)
# Autor: [Tu nombre]
# ==============================================================

# -----------------------------
# 0. CONFIGURACION DEL PROYECTO
# -----------------------------
# El script usa solamente funciones de R base para evitar problemas
# de instalacion de paquetes.

# Crear carpetas de salida
if (!dir.exists("resultados")) dir.create("resultados")
if (!dir.exists("graficas")) dir.create("graficas")

# Cargar datos: primero intenta leer el CSV entregado con el proyecto.
# Si no existe, usa el dataset incorporado en R.
if (file.exists("airquality.csv")) {
  datos_originales <- read.csv("airquality.csv", na.strings = c("NA", ""))
} else {
  data("airquality", package = "datasets")
  datos_originales <- airquality
  write.csv(datos_originales, "airquality.csv", row.names = FALSE, na = "NA")
}

# ==============================================================
# 1. COMPRENSION DEL PROBLEMA / NEGOCIO
# ==============================================================
# Objetivo: estimar la concentracion diaria de ozono (Ozone, ppb)
# a partir de radiacion solar, viento, temperatura y mes.
# Tipo de problema: regresion supervisada.
# Criterio de exito: superar un modelo base (media de entrenamiento)
# y obtener un R2 positivo en datos de prueba.

# ==============================================================
# 2. COMPRENSION DE LOS DATOS
# ==============================================================
cat("\n--- DIMENSIONES ---\n")
print(dim(datos_originales))
cat("\n--- ESTRUCTURA ---\n")
str(datos_originales)
cat("\n--- RESUMEN ---\n")
print(summary(datos_originales))
cat("\n--- VALORES FALTANTES ---\n")
faltantes <- colSums(is.na(datos_originales))
print(faltantes)

# Tabla de faltantes
faltantes_tabla <- data.frame(
  Variable = names(faltantes),
  Faltantes = as.integer(faltantes),
  Porcentaje = round(100 * as.integer(faltantes) / nrow(datos_originales), 2)
)
write.csv(faltantes_tabla, "resultados/faltantes.csv", row.names = FALSE)

# Grafica de faltantes
png("graficas/01_faltantes.png", width = 1000, height = 650, res = 130)
barplot(faltantes,
        main = "Valores faltantes por variable",
        ylab = "Numero de valores faltantes",
        las = 2)
dev.off()

# ==============================================================
# 3. PREPARACION DE LOS DATOS
# ==============================================================
# No se imputa la variable objetivo Ozone: se eliminan esos registros.
datos <- datos_originales[!is.na(datos_originales$Ozone), ]
row.names(datos) <- NULL

# Solar.R tiene algunos faltantes. Se imputan con la mediana calculada
# solamente entre los registros con Ozone disponible.
mediana_solar <- median(datos$Solar.R, na.rm = TRUE)
datos$Solar.R[is.na(datos$Solar.R)] <- mediana_solar

# El mes se usa como variable categorica.
datos$Month_factor <- factor(datos$Month)

cat("\nRegistros originales:", nrow(datos_originales), "\n")
cat("Registros despues de retirar Ozone faltante:", nrow(datos), "\n")
cat("Mediana utilizada para imputar Solar.R:", mediana_solar, "\n")

# Correlaciones de variables numericas principales
correlaciones <- cor(datos[, c("Ozone", "Solar.R", "Wind", "Temp")])
print(round(correlaciones, 3))
write.csv(round(correlaciones, 4), "resultados/correlaciones.csv")

# Resumen de ozono por mes
resumen_mes <- aggregate(Ozone ~ Month, data = datos,
                         FUN = function(x) c(n = length(x), media = mean(x),
                                            mediana = median(x), max = max(x)))
resumen_mes_df <- data.frame(
  Month = resumen_mes$Month,
  n = resumen_mes$Ozone[, "n"],
  media = resumen_mes$Ozone[, "media"],
  mediana = resumen_mes$Ozone[, "mediana"],
  max = resumen_mes$Ozone[, "max"]
)
write.csv(resumen_mes_df, "resultados/ozono_por_mes.csv", row.names = FALSE)

# Graficas exploratorias
png("graficas/02_hist_ozone.png", width = 1000, height = 650, res = 130)
hist(datos$Ozone, breaks = 15,
     main = "Distribucion de Ozone",
     xlab = "Ozone (ppb)")
dev.off()

png("graficas/03_ozone_temp.png", width = 1000, height = 650, res = 130)
plot(datos$Temp, datos$Ozone,
     main = "Ozone vs. temperatura",
     xlab = "Temperatura maxima (F)",
     ylab = "Ozone (ppb)")
abline(lm(Ozone ~ Temp, data = datos), lwd = 2)
dev.off()

png("graficas/04_ozone_wind.png", width = 1000, height = 650, res = 130)
plot(datos$Wind, datos$Ozone,
     main = "Ozone vs. velocidad del viento",
     xlab = "Viento (mph)",
     ylab = "Ozone (ppb)")
abline(lm(Ozone ~ Wind, data = datos), lwd = 2)
dev.off()

png("graficas/05_ozone_mes.png", width = 1000, height = 650, res = 130)
barplot(resumen_mes_df$media,
        names.arg = resumen_mes_df$Month,
        main = "Promedio de Ozone por mes",
        xlab = "Mes",
        ylab = "Ozone promedio (ppb)")
dev.off()

# Division reproducible 80/20 sin depender del generador aleatorio.
# La regla usa el orden de los registros tras la limpieza.
indice <- seq_len(nrow(datos))
es_entrenamiento <- ((indice * 37) %% 10) < 8
entrenamiento <- datos[es_entrenamiento, ]
prueba <- datos[!es_entrenamiento, ]

cat("\nEntrenamiento:", nrow(entrenamiento), "registros\n")
cat("Prueba:", nrow(prueba), "registros\n")

# ==============================================================
# 4. MODELADO
# ==============================================================
# Modelo 1: variables meteorologicas principales
modelo1 <- lm(Ozone ~ Solar.R + Wind + Temp, data = entrenamiento)

# Modelo 2: agrega estacionalidad mediante Month_factor
modelo2 <- lm(Ozone ~ Solar.R + Wind + Temp + Month_factor,
              data = entrenamiento)

cat("\n--- MODELO 1 ---\n")
print(summary(modelo1))
cat("\n--- MODELO 2 ---\n")
print(summary(modelo2))

# ==============================================================
# 5. EVALUACION
# ==============================================================
metricas_regresion <- function(real, predicho) {
  rmse <- sqrt(mean((real - predicho)^2))
  mae <- mean(abs(real - predicho))
  r2 <- 1 - sum((real - predicho)^2) / sum((real - mean(real))^2)
  c(RMSE = rmse, MAE = mae, R2 = r2)
}

pred1 <- predict(modelo1, newdata = prueba)
pred2 <- predict(modelo2, newdata = prueba)

# Modelo base: predecir siempre la media de Ozone en entrenamiento
pred_base <- rep(mean(entrenamiento$Ozone), nrow(prueba))

m_base <- metricas_regresion(prueba$Ozone, pred_base)
m1 <- metricas_regresion(prueba$Ozone, pred1)
m2 <- metricas_regresion(prueba$Ozone, pred2)

metricas <- data.frame(
  Modelo = c("Base: media", "Modelo 1: meteorologico", "Modelo 2: meteorologico + mes"),
  RMSE = c(m_base["RMSE"], m1["RMSE"], m2["RMSE"]),
  MAE = c(m_base["MAE"], m1["MAE"], m2["MAE"]),
  R2 = c(m_base["R2"], m1["R2"], m2["R2"])
)
print(metricas)
write.csv(metricas, "resultados/metricas_modelos.csv", row.names = FALSE)

# Coeficientes del modelo final
coeficientes <- data.frame(
  Termino = names(coef(modelo2)),
  Coeficiente = as.numeric(coef(modelo2))
)
write.csv(coeficientes, "resultados/coeficientes_modelo2.csv", row.names = FALSE)

# Predicciones del conjunto de prueba
predicciones <- data.frame(
  Ozone_real = prueba$Ozone,
  Ozone_predicho = pred2,
  Residuo = prueba$Ozone - pred2
)
write.csv(predicciones, "resultados/predicciones_prueba.csv", row.names = FALSE)

# Real vs. predicho
png("graficas/06_real_predicho.png", width = 900, height = 750, res = 130)
plot(prueba$Ozone, pred2,
     main = "Valores reales vs. predichos (prueba)",
     xlab = "Ozone real (ppb)",
     ylab = "Ozone predicho (ppb)")
abline(0, 1, lty = 2, lwd = 2)
dev.off()

# Residuos
png("graficas/07_residuos.png", width = 1000, height = 650, res = 130)
plot(pred2, prueba$Ozone - pred2,
     main = "Residuos del modelo final",
     xlab = "Ozone predicho (ppb)",
     ylab = "Residuo")
abline(h = 0, lty = 2, lwd = 2)
dev.off()

# ==============================================================
# 6. IMPLEMENTACION / DESPLIEGUE
# ==============================================================
# Funcion sencilla para usar el modelo con nuevos datos.
predecir_ozono <- function(solar_r, wind, temp, month) {
  nuevo <- data.frame(
    Solar.R = solar_r,
    Wind = wind,
    Temp = temp,
    Month_factor = factor(month, levels = levels(entrenamiento$Month_factor))
  )
  as.numeric(predict(modelo2, newdata = nuevo))
}

# Ejemplo de uso
cat("\nEjemplo de prediccion:\n")
cat("Solar.R=200, Wind=8, Temp=85, Month=7 -> Ozone estimado =",
    round(predecir_ozono(200, 8, 85, 7), 2), "ppb\n")

# Guardar el modelo para reutilizarlo sin volver a entrenar
saveRDS(modelo2, "resultados/modelo_ozono.rds")

cat("\nProyecto CRISP-DM terminado. Revisa las carpetas 'graficas' y 'resultados'.\n")
