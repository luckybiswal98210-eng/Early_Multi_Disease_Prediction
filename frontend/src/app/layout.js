import "./globals.css";

export const metadata = {
  title: "Intelligent Healthcare Monitoring & Early Disease Prediction System",
  description: "Non-invasive software-based clinical decision support platform for Parkinson's, Cardiovascular, Diabetes, and Chronic Kidney Disease with multi-user self-registration and MongoDB storage.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
