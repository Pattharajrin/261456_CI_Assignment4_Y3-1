# 261456_CI_Assignment4_Y3-1
Computer Assignment 4

Due Date 31 ตุลาคม 2567 ก่อน 23.00 น. (ไม่มีการเลื่อนส่ง) ส่งผ่าน exam.cmu.ac.th เท่านั้น ห้ามส่งเป็น e-mail หรือเป็นกระดาษ

            จงเขียน program สำหรับการ Train Multilayer Perceptron โดยใช้ Particle Swarm Optimization (PSO) สำหรับการทำ prediction Benzene concentration โดยเป็นการ predict 5 วันล่วงหน้า และ 10 วันล่วงหน้า โดยให้ใช้ attribute เบอร์ 3,6,8,10,11,12,13 และ 14 เป็น input ส่วน desire output เป็น attribute เบอร์ 5   รายงานจะต้องประกอบไปด้วย

            1. ลักษณะการทำงานของระบบ

            2. simulation ของระบบ ผลการทดลอง และวิเคราะห์

            3 โปรแกรม

ให้ทำการทดลองกับ AirQualityUCI (Air Quality Data Set จาก UCI Machine learning Repository) โดยที่ data set นี้มีทั้งหมด 9358 sample และมี 14 attribute ดังนี้

0 Date (DD/MM/YYYY)

1 Time (HH.MM.SS)

2 True hourly averaged concentration CO in mg/m^3 (reference analyzer)

3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)

4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)

5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)

6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)

7 True hourly averaged NOx concentration in ppb (reference analyzer)

8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)

9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)

10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)

11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)

12 Temperature in Â°C

13 Relative Humidity (%)

14 AH Absolute Humidity

ให้ทำการทดลองโดยใช้ 10% cross validation เพื่อทดสอบ validity ของ network ที่ได้ และให้ทำการเปลี่ยนแปลงจำนวน hidden layer และ nodes  

ในการวัด Error ให้ใช้ Mean Absolute Error (MAE)

หมายเหตุ ควรจะเขียนรายงานให้อยู่ในรูปแบบรายงานที่ดี รวมถึงการวิเคราะห์ที่ดีด้วย
