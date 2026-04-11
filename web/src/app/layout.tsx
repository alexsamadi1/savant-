import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "./Sidebar";

export const metadata: Metadata = {
  title: "Savant",
  description: "Organizational knowledge assistant",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Sidebar />
        <main style={{
          marginLeft: 220,
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
        }}>
          {children}
        </main>
      </body>
    </html>
  );
}
