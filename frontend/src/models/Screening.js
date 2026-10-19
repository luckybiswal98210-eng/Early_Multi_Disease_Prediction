import mongoose from 'mongoose';

const ScreeningSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
  },
  userName: {
    type: String,
    required: true,
  },
  userEmail: {
    type: String,
    required: true,
  },
  disease: {
    type: String,
    required: true,
    enum: [
      "Parkinson's Disease",
      "Cardiovascular Disease",
      "Diabetes Mellitus",
      "Chronic Kidney Disease",
      "AI Clinical Prescription",
      "Image Triaging"
    ],
  },
  probability: {
    type: Number,
    required: true,
  },
  status: {
    type: String,
    required: true,
  },
  isDetected: {
    type: Boolean,
    default: false,
  },
  vitals: {
    type: String,
    default: '',
  },
  notes: {
    type: String,
    default: '',
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

export default mongoose.models.Screening || mongoose.model('Screening', ScreeningSchema);
