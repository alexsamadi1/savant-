import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Sidebar from "./Sidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Savant",
  description: "Organizational knowledge assistant for GovCon",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen flex`} style={{ backgroundColor: "#0a0a0f", color: "#e0e0e0" }}>
        <Sidebar />
        <main style={{ marginLeft: 240, flex: 1, minHeight: "100vh" }}>
          {children}
        </main>
      </body>
    </html>
  );
}
