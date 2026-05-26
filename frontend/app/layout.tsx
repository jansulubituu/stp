import type { Metadata } from "next";
import { Plus_Jakarta_Sans, Inter } from "next/font/google";
import "./globals.css";

const displayFont = Plus_Jakarta_Sans({
  subsets: ["latin", "vietnamese"],
  variable: "--font-outfit", // keep the same variable name so globals.css doesn't break
  display: "swap",
});

const inter = Inter({
  subsets: ["latin", "vietnamese"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "PatSight - Patent Intelligence Workspace",
  description: "Hệ thống tra cứu, phân tích và đánh giá bằng sáng chế thông minh dựa trên trí tuệ nhân tạo.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className={`${displayFont.variable} ${inter.variable}`}>
      <body style={{ fontFamily: "var(--font-inter), sans-serif" }}>{children}</body>
    </html>
  );
}
