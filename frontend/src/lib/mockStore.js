// In-memory persistent fallback store if MongoDB Atlas URI is not yet configured by the user
let mockUsers = [
  {
    _id: "usr_admin_1",
    name: "Dr. Admin (Clinic Director)",
    email: "admin@health.org",
    passwordHash: "admin123",
    role: "admin",
    age: 48,
    gender: "Female",
    contact: "+91 98765 00000",
    medicalNotes: "System administrator with full patient oversight.",
    createdAt: new Date("2026-10-01T10:00:00Z").toISOString(),
  },
  {
    _id: "usr_patient_1",
    name: "John Doe (Demo Patient)",
    email: "patient@health.org",
    passwordHash: "patient123",
    role: "patient",
    age: 45,
    gender: "Male",
    contact: "+91 98765 43210",
    medicalNotes: "Baseline multi-system screening profile.",
    createdAt: new Date("2026-10-15T09:30:00Z").toISOString(),
  }
];

let mockScreenings = [
  {
    _id: "scr_1",
    userId: "usr_patient_1",
    userName: "John Doe (Demo Patient)",
    userEmail: "patient@health.org",
    disease: "Diabetes Mellitus",
    probability: 20.9,
    status: "Non-Diabetic / Normal",
    isDetected: false,
    vitals: "Glucose: 108 mg/dL, BMI: 26.4, Age: 45",
    notes: "Normal glycemic response. Routine monitoring advised.",
    timestamp: new Date("2026-10-18T14:20:00Z").toISOString(),
  },
  {
    _id: "scr_2",
    userId: "usr_patient_1",
    userName: "John Doe (Demo Patient)",
    userEmail: "patient@health.org",
    disease: "Cardiovascular Disease",
    probability: 24.1,
    status: "Low Cardiovascular Risk",
    isDetected: false,
    vitals: "BP: 128 mm Hg, Chol: 214 mg/dL, HR: 156 bpm",
    notes: "Hemodynamics within physiological tolerance.",
    timestamp: new Date("2026-10-19T08:15:00Z").toISOString(),
  }
];

export const mockStore = {
  getUsers: () => mockUsers,
  addUser: (user) => {
    mockUsers.unshift(user);
    return user;
  },
  findUserByEmail: (email) => {
    return mockUsers.find((u) => u.email.toLowerCase() === email.toLowerCase());
  },
  getScreenings: (userId = null) => {
    if (userId) {
      return mockScreenings.filter((s) => s.userId === userId);
    }
    return mockScreenings;
  },
  addScreening: (screening) => {
    mockScreenings.unshift(screening);
    return screening;
  }
};
