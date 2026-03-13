# =====================================================================
# Simulate SOC–Decomposer system for multiple parameter sets
# -->Simulate stochastic trajectories for multiple parameter sets.
# =====================================================================
library(deSolve); library(dplyr); library(ggplot2)
library(rstudioapi)

# Get the path of the current script
script_path <- getActiveDocumentContext()$path
dir_path <- dirname(script_path)  # directory where the script is stored

# ---------------------------------------------------------------------
# 1. Define multiple parameter sets
# ---------------------------------------------------------------------
# param_list <- list(
#   list(I=1., d=0.01, e=0.05, m=0.0), #no/no
#   list(I=1., d=0.01, e=0.05, m=0.025),  #no/no
#   list(I=1., d=0.01, e=0.05, m=0.05),  #no/no
#   
#   list(I=1., d=0.01, e=0.05, m=0.025),  #no/no
#   list(I=1., d=0.01, e=0.1, m=0.025),  #no/no
#   list(I=1., d=0.01, e=0.15, m=0.025),  #no/si (y->x)
#   
#   list(I=1., d=0.01, e=0.1, m=0.025), #no/no
#   list(I=5., d=0.01, e=0.1, m=0.025), #si/no (x->y)
#   list(I=10., d=0.01, e=0.1, m=0.025), #si/no (x->y)
#   
#   list(I=5., d=0.01, e=0.15, m=0.0),
#   list(I=5., d=0.01, e=0.15, m=0.025), 
#   list(I=5., d=0.01, e=0.15, m=0.1) 
# )


# I_vals <- c(1, 5)
# e_vals <- c(0.05, 0.5)
# m_vals <- c(0.0001, 1.)
# d_val  <- 0.01

# I_vals <- 1.
# e_vals <- c(0.1, 0.6)
# m_vals <- 1.
# d_vals  <- 0.01
# gam_vals <- 1.

#TEST ELSA
I_vals <- 1.
e_vals <- 0.3
m_vals <- 0.02
d_vals  <- 0.00067
gam_vals <- 1.

# Create grid
param_grid <- expand.grid(
  I = I_vals,
  e = e_vals,
  m = m_vals,
  d = d_vals,
  gam = gam_vals
)


#Function to create list of numeric values from param grid
param_list <- function(param_grid){
  apply(param_grid, 1, function(row) {
  list(
    I = as.numeric(row["I"]),
    d = as.numeric(row["d"]),
    e = as.numeric(row["e"]),
    m = as.numeric(row["m"]),
    gam = as.numeric(row["gam"]),
    alp = as.numeric(row["alp"]),
    bet = as.numeric(row["bet"])
  )
})
}

#Linear model
param_grid_L <- param_grid
param_grid_L$alp <- 0.
param_grid_L$bet <- 1.

param_list_L <- param_list(param_grid_L)


#Multiplicative model
param_grid_M <- param_grid
param_grid_M$alp <- 1.
param_grid_M$bet <- 1.

param_list_M <- param_list(param_grid_M)

#Density dependent mortality model
param_grid_D <- param_grid
param_grid_D$alp <- 1.
param_grid_D$bet <- 2.

param_list_D <- param_list(param_grid_D)


# write.csv(param_grid_L, file.path(dir_path, "param_grid_L.csv"), row.names=FALSE)
# cat("Saved file: param_grid_L\n")
# 
# write.csv(param_grid_M, file.path(dir_path, "param_grid_M.csv"), row.names=FALSE)
# cat("Saved file: param_grid_M\n")
# 
# write.csv(param_grid_D, file.path(dir_path, "param_grid_D.csv"), row.names=FALSE)
# cat("Saved file: param_grid_D\n")

write.csv(param_grid_L, file.path(dir_path, "param_grid_L_testELSA.csv"), row.names=FALSE)
cat("Saved file: param_grid_L\n")

write.csv(param_grid_M, file.path(dir_path, "param_grid_M_testELSA.csv"), row.names=FALSE)
cat("Saved file: param_grid_M\n")

write.csv(param_grid_D, file.path(dir_path, "param_grid_D_testELSA.csv"), row.names=FALSE)
cat("Saved file: param_grid_D\n")


#Function to extract parameter values from list
extract_param_vals<-function(param_list){
  # Extract m values from param_list
  m_values <- sapply(param_list, function(x) x$m)
  # Extract e values from param_list
  e_values <- sapply(param_list, function(x) x$e)
  # Extract I values from param_list
  I_values <- sapply(param_list, function(x) x$I)
  # Extract gam values from param_list
  gam_values <- sapply(param_list, function(x) x$gam)
  # Extract alp values from param_list
  alp_values <- sapply(param_list, function(x) x$alp)
  # Extract bet values from param_list
  bet_values <- sapply(param_list, function(x) x$bet)
  
  # Create labels for m = ..
  m_lab <- setNames(paste0("m = ", m_values,"; e = ", e_values,
                                "; I = ", I_values,"; gam = ", gam_values),
                         seq(1:length(param_list)))
  
  return(list(m=m_values,e=e_values,I=I_values,gam=gam_values,
              alp=alp_values, bet=bet_values, m_labels = m_lab))
}

extract_param_vals_L<-extract_param_vals(param_list_L)
extract_param_vals_M<-extract_param_vals(param_list_M)
extract_param_vals_D<-extract_param_vals(param_list_D)




# ---------------------------------------------------------------------
#init <- c(X=100, Y=5)
init <- c(X=100, Y=1)
times_proc <- seq(0, 100, by=0.01)
n_reps <- 10
set.seed(42)

# ---------------------------------------------------------------------
# 2. ODE system
# ---------------------------------------------------------------------
#X soc
#Y decomposer
# I input
#d decomposition rate const
#e microbial growth efficiency
#m mortality rate
#gam how much necromass is recycled

##################
#Model definition
##################
ode_model <- function(t, state, parms) {
  #Linear if alp = 0; bet = 1
  #Multiplicative if alp=1; bet = 1
  #Density dependent mortality if alp = 1; bet = 2
  with(as.list(c(state, parms)), {
    
    #Model structure parameters
    alp=as.numeric(parms["alp"])
    bet=as.numeric(parms["bet"])
    
    #Varying parameters
    m=as.numeric(parms["m"])
    e=as.numeric(parms["e"])
    I=as.numeric(parms["I"])
    d=as.numeric(parms["d"])
    gam=as.numeric(parms["gam"])
    
    #General model structure
    dX <- I - d*X*Y^alp + gam*m*Y^bet
    dY <- e*d*X*Y^alp - m*Y^bet
    list(c(dX, dY))
  })
}

######################################################
#Function to calculate the steady state
######################################################
ss_model<-function(parms){
  #Model structure parameters
  alp=as.numeric(parms["alp"])
  bet=as.numeric(parms["bet"])
  
  #Varying parameters
  m=as.numeric(parms["m"])
  e=as.numeric(parms["e"])
  I=as.numeric(parms["I"])
  d=as.numeric(parms["d"])
  gam=as.numeric(parms["gam"])
  
  if((1 - gam*e) <= 0) stop("Invalid equilibrium condition: either e or gam needs to be <1")
  
  #NB alpha needs to be >beta
  
  X_star = m/(d*e)*((I/(1-gam*e))*(e/m))^(1-alp/bet)
  Y_star = (e*I/(m*(1-gam*e)))^(1/bet)
  
  
  return(c(X_star,Y_star))
}


######################################################
#Function to calculate the jacobian
######################################################
jacob_model<-function(parms,X_star, Y_star){
  #Model structure parameters
  alp=as.numeric(parms["alp"])
  bet=as.numeric(parms["bet"])
  
  #Varying parameters
  m=as.numeric(parms["m"])
  e=as.numeric(parms["e"])
  I=as.numeric(parms["I"])
  d=as.numeric(parms["d"])
  gam=as.numeric(parms["gam"])
  
  dX_dX = -d*Y_star^alp
  dX_dY = -alp*d*X_star*Y_star^(alp-1)+bet*gam*m*Y_star^(bet-1)
  dY_dX = e*d*Y_star^alp
  dY_dY = alp*e*d*X_star*Y_star^(alp-1)-bet*m*Y_star^(bet-1)
  
  #NB R fills matrices column-wise.
  
  return(matrix(c(dX_dX,dY_dX,dX_dY,dY_dY), nrow = 2,ncol=2))
}


######################################################
#Jacobian evaluated at equlibrium of the linear model
######################################################
J_star_list_L<-list()
for(i in 1:dim(param_grid_L)[1]){
  print(param_grid_L[i,])
  
  params_i <- param_grid_L[i,]
  
  steady_state<- ss_model(params_i)
  
  X_ss = steady_state[1]
  Y_ss = steady_state[2]
  
  J_star = jacob_model(params_i,X_ss,Y_ss)
  
  print(J_star)
  
  J_star_list_L<-append(J_star_list_L,list(J_star))
  
}


# saveRDS(J_star_list_L, file.path(dir_path, "J_star_list_L.rds"))
# cat("Saved file: J_star_list_L\n")

J_star_list_L_save <- do.call(rbind, lapply(seq_along(J_star_list_L), function(i) {
  mat <- J_star_list_L[[i]]
  expand.grid(matrix_id = i, row = 1:nrow(mat), col = 1:ncol(mat)) %>%
    transform(value = as.vector(mat))
}))

#write.csv(J_star_list_L_save, file.path(dir_path, "J_star_list_L_long.csv"), row.names = FALSE)
write.csv(J_star_list_L_save, file.path(dir_path, "J_star_list_L_long_testELSA.csv"), row.names = FALSE)

########################################################################
#Jacobian evaluated at equlibrium of the multiplicative model
########################################################################
J_star_list_M<-list()
for(i in 1:dim(param_grid_M)[1]){
  print(param_grid_M[i,])
  
  params_i <- param_grid_M[i,]
  
  steady_state<- ss_model(params_i)
  
  X_ss = steady_state[1]
  Y_ss = steady_state[2]
  
  J_star = jacob_model(params_i,X_ss,Y_ss)
  
  print(J_star)
  
  J_star_list_M<-append(J_star_list_M,list(J_star))
  
}

# saveRDS(J_star_list_M, file.path(dir_path, "J_star_list_M.rds"))
# cat("Saved file: J_star_list_M\n")

J_star_list_M_save <- do.call(rbind, lapply(seq_along(J_star_list_M), function(i) {
  mat <- J_star_list_M[[i]]
  expand.grid(matrix_id = i, row = 1:nrow(mat), col = 1:ncol(mat)) %>%
    transform(value = as.vector(mat))
}))

#write.csv(J_star_list_M_save, file.path(dir_path, "J_star_list_M_long.csv"), row.names = FALSE)
write.csv(J_star_list_M_save, file.path(dir_path, "J_star_list_M_long_testELSA.csv"), row.names = FALSE)

########################################################################
#Jacobian evaluated at equlibrium of the density-dependent mort model
########################################################################
J_star_list_D<-list()
for(i in 1:dim(param_grid_D)[1]){
  print(param_grid_D[i,])
  
  params_i <- param_grid_D[i,]
  
  steady_state<- ss_model(params_i)
  
  X_ss = steady_state[1]
  Y_ss = steady_state[2]
  
  J_star = jacob_model(params_i,X_ss,Y_ss)
  
  print(J_star)
  
  J_star_list_D<-append(J_star_list_D,list(J_star))
  
}

# saveRDS(J_star_list_D, file.path(dir_path, "J_star_list_D.rds"))
# cat("Saved file: J_star_list_D\n")

J_star_list_D_save <- do.call(rbind, lapply(seq_along(J_star_list_D), function(i) {
  mat <- J_star_list_D[[i]]
  expand.grid(matrix_id = i, row = 1:nrow(mat), col = 1:ncol(mat)) %>%
    transform(value = as.vector(mat))
}))

#write.csv(J_star_list_D_save, file.path(dir_path, "J_star_list_D_long.csv"), row.names = FALSE)
write.csv(J_star_list_D_save, file.path(dir_path, "J_star_list_D_long_testELSA.csv"), row.names = FALSE)



########################################################################
# Euler–Maruyama function to generate synthetic data from model
########################################################################
euler_maruyama <- function(func, init, parms, times, dt=0.01, sigma_proc=c(X=0.5,Y=0.2)){
  nsteps <- length(times)
  state <- as.numeric(init); names(state) <- names(init)
  out <- data.frame(time=times, X=NA_real_, Y=NA_real_)
  out[1,2:3] <- state
  for(i in 2:nsteps){
    tprev <- times[i-1]; tnow <- times[i]
    steps <- ceiling((tnow-tprev)/dt); dt_step <- (tnow-tprev)/steps
    for(s in 1:steps){
      derivs <- func(tprev + (s-1)*dt_step, state, parms)[[1]] #evaluates the deterministic derivative at the current state and time
      state <- state + dt_step*derivs + sqrt(dt_step)*rnorm(length(state), 0, c(sigma_proc["X"], sigma_proc["Y"])) #Euler Maruyama formula
      state <- pmax(state, 0) #avoids negative values
    }
    out[i,2:3] <- state
  }
  out
}

# ---------------------------------------------------------------------
# 3. Loop over parameter sets
# ---------------------------------------------------------------------

#Generate synthetic data with Euler-Maruyama

generat_synth_data<- function(param_list){
  #Num parameter sets
  param_sets<-seq_along(param_list)
  print(" #################### ")
  print(paste0("Generating synthetic data for ", n_reps, " replicates"))
  print(paste0("for " ,length(param_sets), " parameter sets"))
  print(" ")
  print("..It can take a while..")
  print(" #################### ")
  
  all_data <- list()
  for(p_idx in param_sets){
    parms <- param_list[[p_idx]]
    rep_data <- list()
    for(r in 1:n_reps){
      proc_out <- euler_maruyama(ode_model, init, parms, times_proc, 
                                 dt=0.05, #time step for the Euler approximation
                                 sigma_proc=c(X=0.2,Y=0.05) #standard deviations for process noise
      )
      #sample observation times:
      #obs_times <- sort(sample(proc_out$time, size=70)) #Pick 70 random time points from the simulated trajectory to represent observation times.
      obs_times <- proc_out$time[seq(1, length(proc_out$time), by = 5)] #picks every 5th time step
      #adds observation noise
      obs <- proc_out %>% dplyr::filter(time %in% obs_times) %>%
        dplyr::mutate(
          #add noise
          Xobs =  pmax(X + rnorm(n(), 0, 0.01*mean(X)),0), #avoids negative values
          Yobs =  pmax(Y + rnorm(n(), 0, 0.01*mean(Y)),0), #avoids negative values
          rep = r,
          param_set = p_idx
        ) %>% dplyr::select(param_set, rep, time, Xobs, Yobs)
      #store replicate data
      rep_data[[r]] <- obs
    }
    #combine replicates for this parameter set
    all_data[[p_idx]] <- bind_rows(rep_data)
  }
  
  obs_all <- bind_rows(all_data)
  
  long_data <- obs_all %>%
    tidyr::pivot_longer(
      cols = c(Xobs, Yobs),
      names_to = "Pool",
      values_to = "Value"
    )
  
  # Compute mean per Pool and param_set
  # mean_data <- long_data %>%
  #   group_by(param_set, Pool, time) %>%
  #   summarise(mean_value = mean(Value), .groups = "drop")
  
  param_vals<-extract_param_vals(param_list)
  
  # Add m_value to long_data based on param_set
  long_data <- long_data %>%
    mutate(m_value = param_vals$m[param_set])
  
  
  return(list(obs_all, long_data))
}

#######################
# LINEAR MODEL
#######################
gener_data_L<-generat_synth_data(param_list_L)
obs_all_L<-gener_data_L[[1]]
long_data_L<- gener_data_L[[2]]

#Plot
ggplot() +
  # All replicates in faint lines
  geom_line(data = long_data_L, aes(x = time, y = Value, group = interaction(rep, Pool), color = Pool), alpha = 0.3) +
  # Mean trajectory in bold lines
  #geom_line(data = mean_data, aes(x = time, y = mean_value, color = Pool), size = 1.2) +
  facet_wrap(~ param_set, scales = "free_y", labeller = labeller(param_set = extract_param_vals_L$m_labels)) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),  # white plot background
    panel.background = element_rect(fill = "white", color = NA)  # white panel background
  ) +
  labs(
    x = "Time",
    y = "Value",
    color = "Pool",
    title = "LINEAR MODEL"
  )

#ggsave(file.path(dir_path, "timeseries_L.png"))
ggsave(file.path(dir_path, "timeseries_L_testELSA.png"))


#write.csv(obs_all_L, file.path(dir_path, "synthetic_dataset_multiple_paramsets_L.csv"), row.names=FALSE)
write.csv(obs_all_L, file.path(dir_path, "synthetic_dataset_multiple_paramsets_L_testELSA.csv"), row.names=FALSE)
cat("Saved file: synthetic_dataset_multiple_paramsets_L.csv\n")

#######################
# MULTIPLICATIVE MODEL
#######################
gener_data_M<-generat_synth_data(param_list_M)
obs_all_M<-gener_data_M[[1]]
long_data_M<- gener_data_M[[2]]

#Plot
ggplot() +
  # All replicates in faint lines
  geom_line(data = long_data_M, aes(x = time, y = Value, group = interaction(rep, Pool), color = Pool), alpha = 0.3) +
  # Mean trajectory in bold lines
  #geom_line(data = mean_data, aes(x = time, y = mean_value, color = Pool), size = 1.2) +
  facet_wrap(~ param_set, scales = "free_y", labeller = labeller(param_set = extract_param_vals_M$m_labels)) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),  # white plot background
    panel.background = element_rect(fill = "white", color = NA)  # white panel background
  ) +
  labs(
    x = "Time",
    y = "Value",
    color = "Pool",
    title = "MULTIPLICATIVE MODEL"
  )

#ggsave(file.path(dir_path, "timeseries_M.png"))
ggsave(file.path(dir_path, "timeseries_M_testELSA.png"))


#write.csv(obs_all_M, file.path(dir_path, "synthetic_dataset_multiple_paramsets_M.csv"), row.names=FALSE)
write.csv(obs_all_M, file.path(dir_path, "synthetic_dataset_multiple_paramsets_M_testELSA.csv"), row.names=FALSE)
cat("Saved file: synthetic_dataset_multiple_paramsets_M.csv\n")


#######################
# DENSITY DEP MORTALITY MODEL
#######################
gener_data_D<-generat_synth_data(param_list_D)
obs_all_D<-gener_data_D[[1]]
long_data_D<- gener_data_D[[2]]

#Plot
ggplot() +
  # All replicates in faint lines
  geom_line(data = long_data_D, aes(x = time, y = Value, group = interaction(rep, Pool), color = Pool), alpha = 0.3) +
  # Mean trajectory in bold lines
  facet_wrap(~ param_set, scales = "free_y", labeller = labeller(param_set = extract_param_vals_D$m_labels)) +
  theme_minimal() +
  theme(
    plot.background = element_rect(fill = "white", color = NA),  # white plot background
    panel.background = element_rect(fill = "white", color = NA)  # white panel background
  ) +
  labs(
    x = "Time",
    y = "Value",
    color = "Pool",
    title = "DENSITY DEPENDENT MORTALITY MODEL"
  )

#ggsave(file.path(dir_path, "timeseries_D.png"))
ggsave(file.path(dir_path, "timeseries_D_testELSA.png"))


#write.csv(obs_all_D, file.path(dir_path, "synthetic_dataset_multiple_paramsets_D.csv"), row.names=FALSE)
write.csv(obs_all_D, file.path(dir_path, "synthetic_dataset_multiple_paramsets_D_testELSA.csv"), row.names=FALSE)
cat("Saved file: synthetic_dataset_multiple_paramsets_D.csv\n")



#Compute average trajectory across replicates

library(dplyr)

mean_traj_L <- obs_all_L %>%
  group_by(param_set, time) %>%
  summarise(
    X_mean = mean(Xobs),
    Y_mean = mean(Yobs),
    .groups = "drop"
  )

tail(mean_traj_L)


mean_traj_M <- obs_all_M %>%
  group_by(param_set, time) %>%
  summarise(
    X_mean = mean(Xobs),
    Y_mean = mean(Yobs),
    .groups = "drop"
  )

tail(mean_traj_M)

mean_traj_D <- obs_all_D %>%
  group_by(param_set, time) %>%
  summarise(
    X_mean = mean(Xobs),
    Y_mean = mean(Yobs),
    .groups = "drop"
  )

tail(mean_traj_D)
